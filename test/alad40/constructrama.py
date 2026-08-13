#! /usr/bin/env python

import MDAnalysis as mda
import MDAnalysis.analysis.rms
from MDAnalysis.analysis import dihedrals
import pandas as pd
import matplotlib.pyplot as plt


u = mda.Universe("step4_equilibration.crd","prod.xtc")
clp = u.select_atoms("resid 1 and name CLP")
#print(clp)
#print('There are {} residues in the protein'.format(len(protein.residues)))
nl = u.select_atoms("resid 1 and name NL")
#print(nl)
ca = u.select_atoms("resid 1 and name CA")
#print(ca)
crp = u.select_atoms("resid 1 and name CRP")
#print(crp)
protein = u.select_atoms("resname ALAD")
alad_dihe = clp+nl+ca+crp
print(alad_dihe.indices)
alad_list = [alad_dihe]
restrain = mda.analysis.dihedrals.Dihedral(alad_list)
restrain.run()
print(restrain.results.angles)
#rama = dihedrals.Ramachandran(protein,check_protein=False).run()
#print(rama.results.angles.shape)
