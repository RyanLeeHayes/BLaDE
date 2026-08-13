import numpy as np
alf_info={}
alf_info['name']='aladsetup'
alf_info['nsubs']=[2]
alf_info['nblocks']=np.sum(alf_info['nsubs'])
alf_info['ncentral']=0
alf_info['nreps']=1
alf_info['nnodes']=1
alf_info['temp']=298.15
alf_info['enginepath']='/dfs8/rhayes1_lab/bin/CHARMM_EXE/gnu/charmm'
alf_info['lmalf']=True
alf_info['q']=np.loadtxt('prep/q',ndmin=1)
