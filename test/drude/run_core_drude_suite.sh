#!/usr/bin/env bash
# =============================================================================
# Core Drude regression suite (delivery gate)
# =============================================================================
# A minimal, self-contained gate that exercises each core Drude feature once on
# tiny synthetic fixtures. It is the trimmed delivery subset of the full Drude
# test apparatus; the exhaustive sweeps / integration / perf / OpenMM-1:1 suites
# are kept locally outside the BLaDE repo (see README.md).
#
# Usage:
#   OMP_NUM_THREADS=1 ./test/drude/run_core_drude_suite.sh
#   BLADE_EXE=/path/to/blade ./test/drude/run_core_drude_suite.sh
#
# Features covered (one deck each, small fixtures only):
#   1. isotropic self-spring        input_cp1_fd_spring              (FD force)
#   2. anisotropic spring           input_cp7_fd_anisotropic_spring  (FD force)
#   3. Thole screened pair          input_cp7_fd_screened_signal     (FD force)
#   4. NBTHOLE all-4 sub-pairs      input_cp3_nbthole_autobuild_all4_on (energy)
#   5. NBTHOLE 1-4                  input_cp3_fd_nbthole14           (FD force)
#   6. hardwall                     boundary/center minimum-image + too-far fatal
#   7. non-Drude parity smoke       input_cp1_baseline_water216      (optional)
#   8. MSLD + Drude smoke           input_cp5_msld_drude_smoke       (optional)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP_ROOT="${TMP_ROOT:-/tmp/blade}"
RUN_DIR="${RUN_DIR:-${TMP_ROOT}/test_runs/core_drude}"
LOG_DIR="${LOG_DIR:-${TMP_ROOT}/test_logs/core_drude}"
BLADE_EXE="${BLADE_EXE:-${ROOT_DIR}/build/blade}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if [[ ! -x "${BLADE_EXE}" ]]; then
  echo "BLaDE executable not found: ${BLADE_EXE}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}" "${LOG_DIR}"
if [[ ! -e "${RUN_DIR}/test" ]]; then
  ln -s "${ROOT_DIR}/test" "${RUN_DIR}/test"
fi

# --- helpers (lifted from run_cp7_must_suite.sh so this runner is standalone) ---
has_bad_numeric_token() {
  awk '
    BEGIN { IGNORECASE=1; bad=0; }
    /^(IN[0-9]+>|SUBSTITUTE>)/ { next; }
    { if ($0 ~ /(^|[^[:alnum:]_])(nan|inf|infinity)([^[:alnum:]_]|$)/) { bad=1; print; exit; } }
    END { exit (bad?0:1); }
  ' "$1"
}

has_unexpected_error_token() {
  awk '
    BEGIN { IGNORECASE=1; bad=0; }
    /^(IN[0-9]+>|SUBSTITUTE>)/ { next; }
    /Overflow of maxBlockExclCount/ { next; }  # benign: blade self-reallocates when domdecheuristic is off
    /FATAL ERROR:/ { bad=1; print; exit; }
    /illegal memory access|segmentation fault|cudaError|CUDA error|Error in file/ { bad=1; print; exit; }
    /Error:/ { bad=1; print; exit; }
    END { exit (bad?0:1); }
  ' "$1"
}

run_case() {
  local name="$1" input="$2"
  local log="${LOG_DIR}/${name}.log"
  echo "[RUN ] ${name} -> ${input}"
  (cd "${RUN_DIR}" && "${BLADE_EXE}" "${input}" >"${log}" 2>&1)
  if has_bad_numeric_token "${log}" >/dev/null; then
    echo "[FAIL] ${name}: NaN/Inf detected in log (${log})" >&2; exit 1
  fi
  if has_unexpected_error_token "${log}" >/dev/null; then
    echo "[FAIL] ${name}: unexpected runtime error token (${log})" >&2; exit 1
  fi
  echo "[PASS] ${name}"
}

run_expect_fail() {
  local name="$1" input="$2" pattern="$3"
  local log="${LOG_DIR}/${name}.log"
  echo "[RUN ] ${name} -> ${input} (expected failure)"
  set +e
  (cd "${RUN_DIR}" && "${BLADE_EXE}" "${input}" >"${log}" 2>&1)
  local rc=$?
  set -e
  if [[ "${rc}" -eq 0 ]]; then
    echo "[FAIL] ${name}: command unexpectedly succeeded (${log})" >&2; exit 1
  fi
  if ! grep -q "${pattern}" "${log}"; then
    echo "[FAIL] ${name}: missing expected error '${pattern}' (${log})" >&2; exit 1
  fi
  echo "[PASS] ${name}"
}

check_range() {
  local label="$1" value="$2" lower="$3" upper="$4"
  if ! awk -v v="${value}" -v lo="${lower}" -v hi="${upper}" 'BEGIN { exit !(v>=lo && v<=hi) }'; then
    echo "[FAIL] ${label}=${value} not in [${lower}, ${upper}]" >&2; exit 1
  fi
}

count_psf_atoms() { awk '/!NATOM/ { print $1+0; exit }' "$1"; }

fd_stats() {
  awk '
    /ij=/ {
      n++;
      split($0,a,"\\(Emax-Emin\\)/dx="); split(a[2],b,", force=");
      fd=b[1]+0; force=b[2]+0;
      absfd=fd; if (absfd<0) absfd=-absfd;
      absforce=force; if (absforce<0) absforce=-absforce;
      signal=absfd; if (absforce>signal) signal=absforce;
      if (signal>maxsig) maxsig=signal;
      if (signal>=1e-6) nz++;
      diff=fd-force; if (diff<0) diff=-diff;
      if (diff>maxabs) maxabs=diff;
      if (absforce>=1e-3) { rel=diff/absforce; if (rel>maxrel) maxrel=rel; reln++; }
    }
    END {
      if (n==0) { print "0 0 0 0 0 0"; }
      else { if (reln==0) maxrel=0; printf "%d %.8f %.8f %d %.8f %d\n", n, maxabs, maxrel, reln, maxsig, nz; }
    }
  ' "$1"
}

assert_fd_case() {
  local name="$1" input="$2" psf="$3" max_abs_tol="$4" max_rel_tol="$5" min_signal="$6" min_signal_points="$7"
  run_case "${name}" "${input}"
  local log="${LOG_DIR}/${name}.log"
  local n maxabs maxrel reln maxsig nz
  read -r n maxabs maxrel reln maxsig nz <<<"$(fd_stats "${log}")"
  if [[ "${n}" -le 0 ]]; then echo "[FAIL] ${name}: missing finite-difference rows" >&2; exit 1; fi
  local atom_count expected_points
  atom_count="$(count_psf_atoms "${psf}")"
  expected_points=$((3*atom_count))
  if [[ "${n}" -ne "${expected_points}" ]]; then
    echo "[FAIL] ${name}: fdPointCount=${n}, expected=${expected_points} (3*N, includes Drude coordinates)" >&2; exit 1
  fi
  check_range "${name}.maxAbsDiff" "${maxabs}" 0 "${max_abs_tol}"
  check_range "${name}.maxRelDiff" "${maxrel}" 0 "${max_rel_tol}"
  check_range "${name}.maxSignal" "${maxsig}" "${min_signal}" 1000000000
  check_range "${name}.signalPoints" "${nz}" "${min_signal_points}" 1000000000
  echo "[STAT] ${name}: fdPointCount=${n} maxAbsDiff=${maxabs} maxRelDiff=${maxrel} maxSignal=${maxsig} signalPoints=${nz}"
}

drude_stats() {
  awk '
    /DRUDE DIAG>/ { n++; tcom+=$6; trel+=$8; if (($10+0)>maxd) maxd=$10+0; totalHW=$14+0; }
    END {
      if (n==0) { print "0 0 0 0 0"; }
      else { printf "%d %.8f %.8f %.8f %d\n", n, tcom/n, trel/n, maxd+0, totalHW+0; }
    }
  ' "$1"
}

assert_hardwall_short_pair() {
  local label="$1" log="$2" max_dmax="$3"
  local n tcom trel dmax hw
  read -r n tcom trel dmax hw <<<"$(drude_stats "${log}")"
  if [[ "${n}" -le 0 ]]; then
    echo "[FAIL] ${label}: missing DRUDE DIAG samples for hardwall minimum-image check" >&2; exit 1
  fi
  check_range "${label}.hardwallTotal" "${hw}" 0 0
  check_range "${label}.dmaxA" "${dmax}" 0 "${max_dmax}"
  echo "[STAT] ${label}: samples=${n} Tcom=${tcom} Trel=${trel} dmaxA=${dmax} hardwallTotal=${hw}"
}

extract_first_nrg_col() {
  awk -v c="$2" '{ cc=c; if (cc<0) cc=NF+cc+1; if (NF>=cc) { print $cc+0; exit } }' "$1"
}

assert_nrg_col_nonzero() {
  local label="$1" nrg="$2" col="$3" min_abs="$4"
  if [[ ! -s "${nrg}" ]]; then echo "[FAIL] ${label}: missing NRG (${nrg})" >&2; exit 1; fi
  local v vabs
  v="$(extract_first_nrg_col "${nrg}" "${col}")"
  vabs="$(awk -v x="${v}" 'BEGIN { if (x<0) x=-x; printf "%.10f\n", x }')"
  check_range "${label}.abs" "${vabs}" "${min_abs}" 1000000000
  echo "[STAT] ${label}: col=${col} value=${v}"
}

# --- core feature gates ---------------------------------------------------
echo "[INFO] Core Drude gate: finite-difference force checks (all coordinates incl. Drude)"
assert_fd_case "core_spring_isotropic" \
  "test/drude/input_cp1_fd_spring" \
  "${ROOT_DIR}/test/drude/data/drude_2atom.psf" \
  0.500 0.002 0.001 2
assert_fd_case "core_spring_anisotropic" \
  "test/drude/input_cp7_fd_anisotropic_spring" \
  "${ROOT_DIR}/test/drude/data/drude_5atom_aniso.psf" \
  0.500 0.005 0.001 6
assert_fd_case "core_screened_pair" \
  "test/drude/input_cp7_fd_screened_signal" \
  "${ROOT_DIR}/test/drude/data/drude_4atom_bonded.psf" \
  0.120 0.050 0.00001 2
assert_fd_case "core_nbthole14" \
  "test/drude/input_cp3_fd_nbthole14" \
  "${ROOT_DIR}/test/drude/data/drude_6atom_nbthole14_openmm.psf" \
  0.300 0.005 0.001 4

echo "[INFO] Core Drude gate: NBTHOLE all-4 sub-pair autobuild energy"
run_case "core_nbthole_all4" "test/drude/input_cp3_nbthole_autobuild_all4_on"
assert_nrg_col_nonzero \
  "core_nbthole_all4.eedrude" \
  "${RUN_DIR}/cp3_nbthole_autobuild_all4_on.nrg" \
  13 \
  0.000001

echo "[INFO] Core Drude gate: hardwall minimum-image displacement"
run_case "core_hardwall_boundary_short" "test/drude/input_cp7_hardwall_boundary_short"
run_case "core_hardwall_center_short" "test/drude/input_cp7_hardwall_center_short"
assert_hardwall_short_pair "core_hardwall_boundary_short" "${LOG_DIR}/core_hardwall_boundary_short.log" 0.10
assert_hardwall_short_pair "core_hardwall_center_short" "${LOG_DIR}/core_hardwall_center_short.log" 0.10
run_expect_fail \
  "core_hardwall_too_far_expected_fail" \
  "test/drude/input_cp7_hardwall_too_far_expected_fail" \
  "Drude particle moved too far beyond hard wall constraint"

echo "[INFO] Core Drude gate: optional smokes (non-Drude parity, MSLD+Drude)"
run_case "core_nondrude_parity_water216" "test/drude/input_cp1_baseline_water216"
run_case "core_msld_drude_smoke" "test/drude/input_cp5_msld_drude_smoke"

echo "[PASS] Core Drude suite completed"
echo "[INFO] Logs: ${LOG_DIR}"
echo "[INFO] Run directory: ${RUN_DIR}"
