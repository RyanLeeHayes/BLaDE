#! /usr/bin/env python
"""Per-force OpenMM energies on the same CRD frames BLaDE evaluated.

Run AFTER the production run, as its own script; it does no dynamics.
Writes openmm_terms.dat: frame, one column per force (kJ/mol), total.
"""
import glob
import numpy as np
from openmm import *
from openmm.app import *
from openmm.unit import *

# ---------------------------------------------------------------------------
# Paste the system-building part of run_md_simply.py here, unchanged:
# everything from the imports down to (but NOT including) the
# `sim = Simulation(...)` line. It must define `top`, `system` and
# `integrator`. Leave out the reporters, minimization and sim.step calls.
# ---------------------------------------------------------------------------

from openmm.unit import *
from openmm.app import *
from openmm import *
import numpy as np

import mdtraj as md

from HyresBuilder.FFs import *

# force field
ff = "./forcefield"
top_inp = f"{ff}/top_hyres_mix.inp"
param_inp = f"{ff}/param_hyres_mix.inp"
ffparams = CharmmParameterSet(top_inp, param_inp)

# input PDB and PSF file to start a simulation
pdb_file = "demo.pdb"
psf_file = "demo.psf"
pdb = PDBFile(pdb_file)
psf = CharmmPsfFile(psf_file)
top = psf.topology

# simulation parameters
T = 298.0                        # temperature
cutoff = 1.2*unit.nanometer      # nonbonded cutoff
d_switch = 1.1*unit.nanometer    # switch function starting distance
temperature = T*unit.kelvin
salt = 150.0                     # salt concentration in mM
c_ion = salt/1000.0              # salt concentration in M
dt_equil = 0.0001*unit.picoseconds        # time step for equilibration, for bad configuration, use 0.0001 ps
dt_prod =  0.008*unit.picoseconds         # time step for production simulation
prod_step = 100000                        # production steps
equil_step = 10000                        # equilibration steps
log_freq =   10000                        # frequency of log file
traj_freq =  10000                        # frequency of trajectory file
pdb_freq =   10000                        # frequency of pdb_traj file
chk_freq =   10000                        # frequency of checkpoint file
out = 'peptide'



# calculate relative dielectric constant at temperature T in K
def cal_er(T):
    Td = T-273
    er_t = 87.74-0.4008*Td+9.398*10**(-4)*Td**2-1.41*10**(-6)*Td**3
    return er_t
er_t = cal_er(T)                                                   # relative electric constant
er_ref = 60
er = er_t*er_ref/77.6
print(f"relative dielectric constant at temperature {T} K is {er}")


# calculate Debye-Huckel screening length in nm
def cal_dh(c_ion, T):
    NA = 6.02214076e23          # Avogadro's number
    er = cal_er(T)
    lB = 16710/(er*T)          # Bjerrum length in nm, 16710 = e^2/(4*pi*epsilon_0*k_B) in unit of nm*K
    dh = np.sqrt(1/(8*np.pi*lB*NA*1e-24*c_ion))   # Debye-Huckel screening length in nm
    return dh*unit.nanometer
dh = cal_dh(c_ion, T)
print(f"Debye screening length: dh = {dh}")


# create openmm system, no PBC
system = psf.createSystem(ffparams, nonbondedMethod=CutoffNonPeriodic, constraints=HBonds,
                          nonbondedCutoff=cutoff, switchDistance=d_switch, temperature=temperature)

# Debye-Huckel parameters
DH_params = {
        'lmd': 0.00,              # Charge scaling factor of P-Mg interaction, important for RNA, DNA systems, but not for IDPs
        'dh': dh,                 # Debye Huckel screening length
        'er': er,                 # relative dielectric constant
    }

# please check file HyresBuilder/HyresBuilder/FFs.py and look for "def buildSystem".
# this is where you can check all potential energy functions, such as Debye-Huckel electrostatics, H bonds, etc
system = buildSystem(psf, system, DH_params, modification=None)

# MD simulation setup
friction=0.1/unit.picosecond
integrator = LangevinIntegrator(temperature, friction, dt_equil)
plat = Platform.getPlatformByName('CUDA')
prop = {'Precision': 'mixed', 'DeviceIndex': '0'}
sim = Simulation(top, system, integrator, plat, prop)
sim.context.setPositions(pdb.positions)
sim.context.setVelocitiesToTemperature(temperature)
print(f'Langevin, CUDA, {temperature}')



def read_crd(fn):
    """CHARMM CRD by fixed columns (CharmmCrdFile splits on whitespace, which
    breaks when a coordinate fills its whole field, e.g. 5013.26660)."""
    lines = open(fn).read().splitlines()
    i = 0
    while lines[i].startswith("*"):
        i += 1
    head = lines[i].split()
    n = int(head[0])
    if len(head) > 1 and head[1].upper() == "EXT":   # extended: 3F20.10 from column 41
        cols = (slice(40, 60), slice(60, 80), slice(80, 100))
    else:                                            # standard: 3F10.5 from column 21
        cols = (slice(20, 30), slice(30, 40), slice(40, 50))
    xyz = np.array([[float(l[c]) for c in cols] for l in lines[i + 1:i + 1 + n]])
    if len(xyz) != system.getNumParticles():
        raise SystemExit(f"{fn}: {len(xyz)} atoms, system has {system.getNumParticles()}")
    return xyz * 0.1 * nanometer                     # A -> nm


# 1. One force group per force. This must happen before the Context exists.
names = []
for i, f in enumerate(system.getForces()):
    f.setForceGroup(i)
    names.append(f"{i}:{f.getName().replace(' ', '_')}")
    print(i, f.getName(), type(f).__name__)

hb = system.getForce(10)
print("donors", hb.getNumDonors(), "acceptors", hb.getNumAcceptors())
don = [hb.getDonorParameters(i)[:2] for i in range(hb.getNumDonors())]   # (N, H), check ordering
acc = [hb.getAcceptorParameters(i)[0] for i in range(hb.getNumAcceptors())]
print(don[:5], don[-5:]); print(acc[-6:])
# 2. A Context only for evaluating energies (any integrator will do)
plat = Platform.getPlatformByName('CUDA')
prop = {'Precision': 'mixed', 'DeviceIndex': '0'}
sim = Simulation(top, system, VerletIntegrator(0.001), plat, prop)

# 3. The same CRD files BLaDE read, in the same order
frames = sorted(glob.glob("testcoords/frame_*.crd"))
if not frames:
    raise SystemExit("no testcoords/frame_*.crd files found")

with open("openmm_terms.dat", "w") as out:
    out.write("# frame " + " ".join(names) + " total   (kJ/mol)\n")
    for k, fn in enumerate(frames):
        sim.context.setPositions(read_crd(fn))
        e = [sim.context.getState(getEnergy=True, groups={i}).getPotentialEnergy()
                .value_in_unit(kilojoule_per_mole) for i in range(len(names))]
        tot = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        out.write(f"{k:5d} " + " ".join(f"{x:14.6f}" for x in e) + f" {tot:14.6f}\n")
print(f"wrote openmm_terms.dat ({len(frames)} frames)")
