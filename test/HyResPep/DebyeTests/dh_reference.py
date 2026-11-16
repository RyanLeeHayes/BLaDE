"""Analytic reference energies (kJ/mol) for the Debye-Huckel test systems.
Matches the HyRes OpenMM forms: nbdirect uses the OpenMM quintic switch,
1-4 pairs use bare screened Coulomb plus plain LJ."""
import numpy as np

KE  = 138.935456   # kJ mol^-1 nm e^-2 (OpenMM value)
ER  = 60.0         # relative dielectric used in the energy  -- set to your BLaDE value
DH  = 0.783        # Debye length in nm                     -- set to your BLaDE value
RSW, RC = 1.6, 1.8 # switch and cutoff, nm

def switch(r):
    if r <= RSW: return 1.0
    if r >= RC:  return 0.0
    x = (r - RSW) / (RC - RSW)
    return 1 - 10*x**3 + 15*x**4 - 6*x**5

def dh(qq, r):                 # bare Debye-Huckel
    return KE / ER * qq * np.exp(-r / DH) / r

def dh_switched(qq, r):        # nbdirect form
    return dh(qq, r) * switch(r)

def lj(eps, rmin, r):          # CHARMM Rmin form, eps > 0 here
    return eps * ((rmin / r)**12 - 2 * (rmin / r)**6)

print("Point-charge pair (q1q2 = -1)")
print(f"{'r (A)':>8} {'E_bare':>14} {'E_switched':>14}")
dists = list(range(2, 21)) + [16.25, 16.5, 16.75, 17.25, 17.5, 17.75]
for d in sorted(dists):
    r = d / 10
    print(f"{d:8.2f} {dh(-1, r):14.8f} {dh_switched(-1, r):14.8f}")

print("\n5-bead chain (planar trans)")
xyz = np.array([[0, 0, 0], [3.1128, 2.1796, 0], [6.2256, 0, 0],
                [9.3384, 2.1796, 0], [12.4512, 0, 0]]) / 10   # nm
q = [1, -1, 1, -1, 1]
eps = 0.1 * 4.184                 # kJ/mol; CHARMM eps_ij = sqrt(eps_i eps_j)
rmin = 0.2 + 0.2                  # nm; Rmin_ij = Rmin/2_i + Rmin/2_j
for (i, j, kind) in [(0, 3, "nb14"), (1, 4, "nb14"), (0, 4, "nbdirect")]:
    r = np.linalg.norm(xyz[i] - xyz[j])
    if kind == "nb14":
        e_el, e_lj = dh(q[i]*q[j], r), lj(eps, rmin, r)
    else:
        e_el = dh_switched(q[i]*q[j], r)
        e_lj = float("nan")  # depends on BLaDE vdwMethod; compare separately
    print(f"A{i+1}-A{j+1} {kind:9s} r={r*10:7.4f} A  E_elec={e_el:12.8f}  E_LJ={e_lj:12.8f}")
