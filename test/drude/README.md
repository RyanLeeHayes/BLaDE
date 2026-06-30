# BLaDE Drude - core test suite

This directory holds the **core** Drude regression tests shipped with BLaDE: a
minimal set that exercises each Drude feature once on tiny synthetic fixtures.

## Run

```bash
# from the BLaDE repo root, with a built ./build/blade
OMP_NUM_THREADS=1 ./test/drude/run_core_drude_suite.sh
# or point at any blade binary:
BLADE_EXE=/path/to/blade ./test/drude/run_core_drude_suite.sh
```

The runner is self-contained (no external Python/OpenMM needed) and asserts a
finite-difference force match or single-point energy for each feature.

## What it covers

| # | feature | deck | driver |
|---|---|---|---|
| 1 | isotropic self-spring        | `input_cp1_fd_spring`                 | FD force |
| 2 | anisotropic spring (3-axis)  | `input_cp7_fd_anisotropic_spring`     | FD force |
| 3 | Thole screened pair `S(u)`   | `input_cp7_fd_screened_signal`        | FD force |
| 4 | NBTHOLE all-4 sub-pairs      | `input_cp3_nbthole_autobuild_all4_on` | energy   |
| 5 | NBTHOLE 1-4                  | `input_cp3_fd_nbthole14`              | FD force |
| 6 | hardwall (minimum image)     | `input_cp7_hardwall_{boundary,center}_short` | dynamics |
| 7 | non-Drude parity (optional)  | `input_cp1_baseline_water216`         | dynamics |
| 8 | MSLD + Drude (optional)      | `input_cp5_msld_drude_smoke`          | dynamics |

Fixtures live in `data/` (~13 small synthetic CHARMM PSF/CRD + one parameter
patch `par_cp3_patch.prm`). Stock force-field files are read from the standard
`test/toppar*` trees.

## Where the rest of the tests are

This is only the delivery core. The **exhaustive** Drude validation (the full
cp1-cp7 sweeps, integration/perf suites, the real 200 mM NaCl polarizable
system, finite-difference/PME variants, restart chains, and the
`compare_cp3_openmm_reference.py` cross-engine harness) is kept **locally,
outside this repo**, under `test4/tests/drude_native/` (a faithful mirror of the
original layout). The **bit-1:1-vs-OpenMM** cross-engine tests live under
`test4/tests/openmm_1to1/` (run via `test4/tests/run_all.py`).

See `doc/DRUDE.txt` for the command reference and known limitations.
