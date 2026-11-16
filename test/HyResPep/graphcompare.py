#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


BladeE = np.loadtxt("compare.nrg")
potE = BladeE[:,-1]
potE*=4.184 #change from kcal/mol to kJ/mol

openE = np.loadtxt("peptide.log",delimiter=",")
opotE = openE[:,1]


fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.plot(potE,opotE,color='mediumblue')
ax.set_xlabel("BLaDE potential energy(kJ/mol)")
ax.set_ylabel("openMM HyRes potential energy(kJ/mol)")
ax.set_title("Force Matching between BLaDE implementation of HyRes Potentials vs. openMM reference")
#ax.fill_between(bin_center_i,center_f_i+center_df_i,center_f_i-center_df_i,
#        alpha=0.5, edgecolor='dodgerblue', facecolor='dodgerblue')
#ax.plot(bin_center_i, center_f_i2, color='red')
#ax.fill_between(bin_center_i,center_f_i2+center_df_i2,center_f_i2-center_df_i2,
#        alpha=0.5, edgecolor='salmon', facecolor='salmon')
fig.savefig("HyRes Compare.pdf",bbox_inches='tight')

