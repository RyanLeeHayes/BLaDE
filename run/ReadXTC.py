import numpy as np
import sys

import MDAnalysis as mda
# See https://docs.python.org/2/tutorial/modules.html for instructions on submodules:
# import MDAnalysis.analysis.distances
# from MDAnalysis.analysis import contacts


PDB='../test/T4L_prep/minimized.pdb'
XTC='test.xtc'
traj=mda.Universe(PDB,format="PDB")
traj.load_new(XTC,format="XTC")
n_frames=traj.trajectory.n_frames

# angles=np.zeros((n_frames,))
# SelectionAng=traj.select_atoms("(resid "+str(i+1)+":"+str(i+4)+") and (name P or name PA)")
# print(SelectionAng)
# for f in range(0,n_frames):
#   # frame=traj[i].trajectory[f]
#   traj.trajectory[f]
#   angles[f]=SelectionAng.dihedral.dihedral()


# np.savetxt("pseudodih/pseudodih_"+str(T)+"_"+str(i)+".dat",angles)
