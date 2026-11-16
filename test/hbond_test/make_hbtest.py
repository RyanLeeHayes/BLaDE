"""Coordinates, reference values and BLaDE input for the HyRes H-bond test.

Run:  python3 make_hbtest.py
Writes:
  crd/*.crd          one CHARMM CRD per geometry (atom order N1 H1 N2 H2 C1 O1 C2 O2)
  hbtest_ref.txt     reference H-bond energy (kJ/mol) per geometry, and the
                     gradient (kJ/mol/nm) for the geometries used in run test spatial
  hbtest.inp         BLaDE input: every geometry with the domdec builder, then brute

Energy, OpenMM HyRes CustomHbondForce form, over all donor x acceptor pairs:
  E = eps*(5 s^12 - 6 s^10) * step(-cos^3 phi) * (-cos^3 phi),  s = sigma/r
  r = |O - N| (minimum image), phi = O-H-N angle with the vertex at H, cutoff on r
"""
import os
import numpy as np

SIGMA = 2.9          # A
EPS = 2.2 * 4.184    # kJ/mol
RCUT = 4.5           # A
BOX = 40.0           # A, cubic
BNH, BCO = 1.00, 1.23  # bond lengths, equal to the b0 values in hbtest.prm

NAMES = [("DON", 1, "DNR", "N"), ("DON", 1, "DNR", "H"),
         ("DON", 2, "DNR", "N"), ("DON", 2, "DNR", "H"),
         ("ACC", 1, "ACR", "C"), ("ACC", 1, "ACR", "O"),
         ("ACC", 2, "ACR", "C"), ("ACC", 2, "ACR", "O")]
DONORS = [(0, 1), (2, 3)]   # (N, H)
ACCEPTORS = [5, 7]          # O


def pair_geometry(r, phi_deg):
    """N, H, O, C for one H-bond with |O-N| = r and O-H-N angle phi, N at the origin."""
    phi = np.radians(phi_deg)
    N = np.zeros(3)
    H = np.array([BNH, 0.0, 0.0])
    u = np.array([-np.cos(phi), np.sin(phi), 0.0])           # unit vector H->O
    d = BNH * np.cos(phi) + np.sqrt((BNH * np.cos(phi))**2 - BNH**2 + r**2)
    O = H + d * u
    C = O + BCO * u                                          # carbonyl C beyond O
    return N, H, O, C


def rotation(seed):
    a, b, c = np.random.default_rng(seed).uniform(0, 2 * np.pi, 3)
    Rz = lambda t: np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    Rx = lambda t: np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    return Rz(a) @ Rx(b) @ Rz(c)


def assemble(N1, H1, O1, C1, N2, H2, O2, C2):
    return np.array([N1, H1, N2, H2, C1, O1, C2, O2])


# Pair 2 parked well away from pair 1 and from each other (all cross distances >> 4.5 A)
PARK_DON = (np.array([28.0, 10.0, 10.0]), np.array([29.0, 10.0, 10.0]))
PARK_ACC = (np.array([10.0, 28.0, 28.0]), np.array([10.0, 29.23, 28.0]))  # O, C


def single_pair(r, phi, center=(10.0, 10.0, 10.0), rot=None):
    N, H, O, C = pair_geometry(r, phi)
    pts = np.array([N, H, O, C])
    if rot is not None:
        pts = pts @ rot.T
    pts = pts + np.array(center)
    N, H, O, C = pts
    return assemble(N, H, O, C, PARK_DON[0], PARK_DON[1], PARK_ACC[0], PARK_ACC[1])


def mic(d):
    return d - BOX * np.round(d / BOX)


def energy_grad(x):
    """Total H-bond energy (kJ/mol) and gradient (kJ/mol/A), minimum image, all pairs."""
    E = 0.0
    G = np.zeros_like(x)
    npairs = 0
    for (n, h) in DONORS:
        for o in ACCEPTORS:
            dr = mic(x[o] - x[n])
            r = np.linalg.norm(dr)
            a = mic(x[o] - x[h])
            b = mic(x[n] - x[h])
            ai, bi = 1 / np.linalg.norm(a), 1 / np.linalg.norm(b)
            c = a @ b * ai * bi
            if r < RCUT:
                npairs += 1
            if not (r < RCUT and c < 0):
                continue
            s2 = (SIGMA / r)**2
            s10 = s2**5
            s12 = s10 * s2
            U = EPS * (5 * s12 - 6 * s10)
            dU = 60 * EPS * (s10 - s12) / r
            g = -c**3
            dg = -3 * c * c
            fr = g * dU / r
            fc = U * dg
            gO = fr * dr + fc * ai * (b * bi - c * a * ai)
            gN = -fr * dr + fc * bi * (a * ai - c * b * bi)
            G[o] += gO
            G[n] += gN
            G[h] += -(gO + gN)
            E += U * g
    return E, G, npairs


def write_crd(path, x, title):
    xw = x % BOX   # wrapped into [0, BOX), as BLaDE does at a domdec reset
    with open(path, "w") as f:
        f.write(f"* {title}\n*\n")
        f.write(f"{len(xw):5d}\n")
        for i, ((seg, resid, resn, name), p) in enumerate(zip(NAMES, xw)):
            f.write(f"{i+1:5d}{resid:5d} {resn:<4s} {name:<4s}"
                    f"{p[0]:10.5f}{p[1]:10.5f}{p[2]:10.5f} {seg:<4s} {resid:<4d}{0.0:10.5f}\n")
    return xw


geoms = []
# 1. Distance scan, linear H-bond (phi = 180); 4.60 A is beyond the cutoff
for r in [2.6, 2.8, 2.9, 3.0, 3.2, 3.5, 4.0, 4.4, 4.6]:
    geoms.append((f"dist_{r:.2f}", single_pair(r, 180.0), False))
# 2. Angle scan at r = 2.9 A; 80 deg has cos > 0, so zero energy
for phi in [180, 165, 150, 135, 120, 105, 95, 80]:
    geoms.append((f"ang_{phi:03d}", single_pair(2.9, phi), False))
# 3. Bent, stretched and rotated out of plane: the spatial-test geometry
geoms.append(("bent", single_pair(3.1, 150.0, rot=rotation(7)), True))
# 4. Two donors and two acceptors in an antiparallel ladder (like a beta-sheet pair):
#    N1-H1...O2 and N2-H2...O1 are H-bonds; N1..O1 and N2..O2 are inside the cutoff
#    but bent the wrong way (cos > 0), so the list must hold all 4 pairs and 2 score zero
lad = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],      # N1 H1
                [3.0, 3.0, 0.3], [2.0, 3.0, 0.3],      # N2 H2
                [-1.23, 3.0, 0.3], [0.0, 3.0, 0.3],    # C1 O1
                [4.13, 0.0, 0.0], [2.9, 0.0, 0.0]])    # C2 O2
geoms.append(("multi", lad @ rotation(11).T + np.array([10.0, 10.0, 10.0]), True))
# 5. Pair straddling the periodic boundary in x
geoms.append(("pbc", single_pair(2.9, 170.0, center=(BOX - 1.4, 10.0, 10.0)), True))

os.makedirs("crd", exist_ok=True)
lines = ["# geometry   E_hbond(kJ/mol)   donor-acceptor pairs with |O-N| < cutoff",
         "# Gradients below are in kJ/mol/nm, the units of BLaDE's force column in run test"]
grads = []
for name, x, show in geoms:
    xw = write_crd(f"crd/{name}.crd", x, f"H-bond test geometry {name}")
    E, G, n = energy_grad(xw)
    lines.append(f"{name:10s} {E:16.8f} {n:4d}")
    if show:
        grads.append((name, G * 10.0))   # kJ/mol/A -> kJ/mol/nm

out = lines[:]
for name, G in grads:
    out.append(f"\n# gradient for {name} (kJ/mol/nm); atom order N1 H1 N2 H2 C1 O1 C2 O2")
    for i, g in enumerate(G):
        out.append(f"{i+1:3d} {NAMES[i][3]:2s} {g[0]:14.6f} {g[1]:14.6f} {g[2]:14.6f}")
open("hbtest_ref.txt", "w").write("\n".join(out) + "\n")

# BLaDE input: every geometry with the domdec builder, then again with brute force
header = """parameters file hbtest.prm
structure file psf hbtest.psf
coordinates file crd crd/dist_2.90.crd
msld nblocks 0
coordinates box cubi {box} {box} {box} 90 90 90
run setvariable elecmethod fswitch
run setvariable rcut 10
run setvariable rswitch 8
run setvariable hbond on
run setterm bond off
run setterm angle off
run setterm nb14 off
run setterm nbdirect off
""".format(box=f"{BOX:g}")
body = []
for builder in ["domdec", "brute"]:
    body.append(f"run setvariable hbondlist {builder}")
    for name, _, _ in geoms:
        body.append(f"coordinates file crd crd/{name}.crd")
        body.append("run energy")
open("hbtest.inp", "w").write(header + "\n".join(body) + "\n")

print("\n".join(lines))
