#!/bin/bash
#source env-charmm                                          
 #source /dfs8/rhayes1_lab/cyeh5/ALF/env-alf/bin/activate
  
python -c "import alf; alf.initialize(engine='blade')"
python -c "import alf; alf.runflat(1,1,13000,39000,engine='blade',G_imp='`pwd`/G_imp',ntersite=[0,0])"
#python -c "import alf; alf.runflat(301,330,125000,375000,engine='bladelib',G_imp='`pwd`/G_imp',ntersite=[0,0])" 
