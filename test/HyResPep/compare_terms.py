#! /usr/bin/env python
"""Term-by-term BLaDE vs OpenMM on identical frames.

compare.nrg       BLaDE run energy rows (kcal/mol)
openmm_terms.dat  output of openmm_terms.py (kJ/mol)
"""
import numpy as np

KCAL = 4.184
TERMS = ["bond", "angle", "urey", "dihe", "impr", "cmap", "nb14", "nbdirect",
         "nbrecip", "nbrecipself", "nbrecipexcl", "lambda", "theta", "cats",
         "noe", "harmonic", "mmfp", "resd", "mlp", "hbond", "bias",
         "potential", "kinetic", "total"]

# OpenMM force index -> BLaDE column(s). Indices are from the force-group printout:
#  0 HarmonicBondForce   1 HarmonicBondForce   2 PeriodicTorsionForce
#  3 CustomTorsionForce  4 CMAPTorsionForce    5 LJ Force w/ NBFIX
#  6 CMMotionRemover     7 ReBAngleForce       8 DH_ElecForce
#  9 1-4 interaction    10 N-H--O HBForce
MAP = [("bond",      [0],    ["bond"]),
       ("urey",      [1],    ["urey"]),      # check: is force 1 the Urey-Bradley term?
       ("angle",     [7],    ["angle"]),
       ("dihe",      [2],    ["dihe"]),
       ("impr",      [3],    ["impr"]),
       ("cmap",      [4],    ["cmap"]),
       ("nb14",      [9],    ["nb14"]),
       ("nbdirect",  [5, 8], ["nbdirect"]),  # LJ + Debye-Hueckel
       ("hbond",     [10],   ["hbond"]),
       ("potential", None,   ["potential"])]

b = np.loadtxt("compare.nrg") * KCAL          # step column is scaled too, but unused
o = np.loadtxt("openmm_terms.dat")
if len(b) != len(o):
    raise SystemExit(f"frame count mismatch: BLaDE {len(b)} vs OpenMM {len(o)}")
bcol = lambda t: b[:, 1 + TERMS.index(t)]

print(f"{'term':10s} {'OpenMM mean':>13s} {'BLaDE mean':>13s} {'mean diff':>11s} "
      f"{'std diff':>10s} {'max|diff|':>10s}")
for label, oidx, bterms in MAP:
    ov = o[:, -1] if oidx is None else o[:, [1 + i for i in oidx]].sum(axis=1)
    bv = sum(bcol(t) for t in bterms)
    d = bv - ov
    print(f"{label:10s} {ov.mean():13.4f} {bv.mean():13.4f} {d.mean():11.4f} "
          f"{d.std(ddof=1):10.4f} {np.abs(d).max():10.4f}")
