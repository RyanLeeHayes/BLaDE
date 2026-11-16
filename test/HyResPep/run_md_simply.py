from openmm.unit import *
from openmm.app import *
from openmm import *
import numpy as np

import mdtraj as md

from HyresBuilder.FFs import *
import parmed as pmd
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
prod_step = 200000                        # production steps
equil_step = 10000                        # equilibration steps
log_freq =   2000                        # frequency of log file
traj_freq =  2000                        # frequency of trajectory file
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

class DHReporter:
    def __init__(self, file, reportInterval):
        self.file = file
        self.reportInterval = reportInterval

        # Open file for writing

        self._out = open(file,"w")

        #Header
        self._out.write(
            "# STEP TIME(ps) DH_Electrostatic_Energy(kJ/mol)\n"
        )
    def describeNextReport(self, simulation):
        steps = self.reportInterval - simulation.currentStep % self.reportInterval

        return {
            "steps":steps,
            "include":["energy"],
            "groups": {8},
                }
    def report(self, simulation, state):
        dh_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

        time = state.getTime().value_in_unit(unit.picoseconds)

        self._out.write(
            f"{simulation.currentStep:8d} "
            f"{time:12.4f} "
            f"{dh_energy:20.10f}\n"
        )

        self._out.flush()

    def __del__(self):
        if hasattr(self, "_out"):
            self._out.close()
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

for i, f in enumerate(system.getForces()):
    f.setForceGroup(i)
    print(i, f.getName(), type(f).__name__)
# MD simulation setup
friction=0.1/unit.picosecond
integrator = LangevinIntegrator(temperature, friction, dt_equil)
plat = Platform.getPlatformByName('CUDA')
prop = {'Precision': 'mixed', 'DeviceIndex': '0'}
sim = Simulation(top, system, integrator, plat, prop)
sim.context.setPositions(pdb.positions)
sim.context.setVelocitiesToTemperature(temperature)
print(f'Langevin, CUDA, {temperature}')



################### Minimization, Equilibriation, Production simulation ####################'
print('\n# Minimization running:')
print('Potential energy before: ', sim.context.getState(getEnergy=True).getPotentialEnergy())
sim.minimizeEnergy(maxIterations=500000, tolerance=0.01)
print('Potential energy after: ', sim.context.getState(getEnergy=True).getPotentialEnergy())
state = sim.context.getState(getPositions=True)
positions = state.getPositions()
structure = pmd.load_file(psf_file)
structure.coordinates = positions.value_in_unit(unit.angstrom)
structure.save("minimized.crd", format="charmmcrd", overwrite=True)
print('\n# Equilibriation running:')
sim.step(equil_step)
state = sim.context.getState(getPositions=True)
positions = state.getPositions()
structure = pmd.load_file(psf_file)
structure.coordinates = positions.value_in_unit(unit.angstrom)
structure.save("equilibrated.crd", format="charmmcrd", overwrite=True)
## save a pdb traj using large step, xtc/dcd traj using small step, and log file
sim.reporters.append(PDBReporter(f'{out}.pdb', pdb_freq))
#sim.reporters.append(XTCReporter(f'{out}.xtc', traj_freq))      # xtc traj
sim.reporters.append(DCDReporter(f'{out}.dcd', traj_freq))      # dcd traj
sim.reporters.append(StateDataReporter(f'{out}.log', log_freq, progress=True, totalSteps=prod_step, step=True, temperature=True, potentialEnergy=True, totalEnergy=True, speed=True))
sim.reporters.append(CheckpointReporter(f'{out}.chk', chk_freq))
sim.reporters.append(DHReporter("dh_energy.dat",reportInterval=2000))
print('\n# Production simulation running:')
sim.integrator.setStepSize(dt_prod)
sim.step(prod_step)
# ... create sim as before, then for each frame:
sim.context.setPositions(positions)   # ideally read back from the same CRD files
terms = {f.getName(): sim.context.getState(getEnergy=True, groups={i}).getPotentialEnergy()
                         .value_in_unit(unit.kilojoule_per_mole)
         for i, f in enumerate(system.getForces())}
sim.saveCheckpoint(f'{out}.chk')
print('\n# Finished!')
