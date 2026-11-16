#! /bin/bash 

#module load cuda/13.0.1
source /opt/apps/anaconda/2024.06/etc/profile.d/conda.sh
conda activate "/data/homezvol0/cyeh5/.conda/envs/openmm-test"
python openmm_terms.py 
