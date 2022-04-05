#!/bin/bash
#    This is a PBS file for gollum

ii=$SLURM_ARRAY_TASK_ID
ibeg=$(( ( $ii - 1 ) * $nitt + 1 ))
iend=$(( $ii * $nitt ))

module load anaconda/2020.07
source charmm.sh

DIR=`pwd`

name=`cat name`
nnodes=`cat nnodes`
nreps=`cat nreps`

RUNDIR=$DIR/run$i

# If this is the first iteration, set up the run directory
if [ $ii -eq 1 ]
then

mkdir $RUNDIR
mkdir $RUNDIR/res $RUNDIR/dcd $RUNDIR/failed

cp variables$ini.inp $RUNDIR/variablesprod.inp
ln -s `pwd`/prep $RUNDIR/prep

fi
# Done setting up the run directory

cd $RUNDIR

while [ $ibeg -gt 1 -a `../ALF/GetSteps.py res/${name}_prod$(( $ibeg - 1 )).lmd_0` -ne 50000 ]
do
echo "Error: Run $ibeg incomplete. Going back one step"
ibeg=$(( $ibeg - 1 ))
done

for itt in `seq $ibeg $iend`
do

# Keep trying until simulation completes correctly with 50000 steps in lambda file
while [ `../ALF/GetSteps.py res/${name}_prod${itt}.lmd_0` -ne 50000 ]
do

# If run failed in past, file the failed files before moving on
if [ -f output_${itt} ]; then
  cp output_${itt} output_${itt}_* error_${itt}* failed/
  rm output_${itt} output_${itt}_* error_${itt}*
fi


# Run the simulation
echo "variables set nsteps 500000" > arguments.inp
echo "variables set itt $itt" >> arguments.inp
export OMP_NUM_THREADS=$nnodes
time $CHARMMEXEC ../msld_prod.inp > output_$itt 2> error_$itt

done

done
