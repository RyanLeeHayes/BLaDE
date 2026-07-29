#!/usr/bin/env bash
set -euo pipefail

blade=${1:-./blade}
test_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT

sed \
  -e "s|^variables set dir .*|variables set dir $test_dir|" \
  -e "s|^run setvariable fnmxtc .*|run setvariable fnmxtc $tmp/sdmd.xtc|" \
  -e "s|^run setvariable fnmlmd .*|run setvariable fnmlmd $tmp/sdmd.lmd|" \
  -e "s|^run setvariable fnmnrg .*|run setvariable fnmnrg $tmp/sdmd.nrg|" \
  -e "s|^run setvariable fnmcpo .*|run setvariable fnmcpo $tmp/before.cpt|" \
  "$test_dir/minimize_sdmd.inp" > "$tmp/test.inp"

printf '%s\n' \
  "run setvariable fnmcpi $tmp/before.cpt" \
  "run setvariable fnmcpo $tmp/after.cpt" \
  "run setvariable nsteps 1" \
  "run setvariable dxatommax 0" \
  "run minimize" >> "$tmp/test.inp"

set +e
"$blade" "$tmp/test.inp" > "$tmp/test.log" 2>&1
status=$?
set -e

[[ $status -eq 1 ]]
awk '
  $1 == "MINI>" && $2 ~ /^[0-9]+$/ {
    if ($3 !~ /^[-+]?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$/ ||
        (count && $3 > previous)) bad=1
    previous=$3
    count++
  }
  END { exit bad || count != 80 }
' "$tmp/test.log"
[[ $(awk '/SDMD> Rejected invalid scaling/ { count++ } END { print count+0 }' "$tmp/test.log") -eq 10 ]]
grep -q 'SDMD> Failed to find a finite non-increasing step after 10 attempts; restored step 0' "$tmp/test.log"
grep -q 'SDMD minimization failed after restoring the last accepted state' "$tmp/test.log"
if (( ${OMP_NUM_THREADS:-1} > 1 )); then
  grep -Eq '^Device [0-9]+ (can|cannot) access device [0-9]+ directly' "$tmp/test.log"
fi
grep -q '^Position ' "$tmp/before.cpt"
grep -q '^Position ' "$tmp/after.cpt"
cmp -s \
  <(sed -n '/^Position /,/^Velocity /p' "$tmp/before.cpt" | sed '$d') \
  <(sed -n '/^Position /,/^Velocity /p' "$tmp/after.cpt" | sed '$d')
