#!/bin/bash
#    This is a PBS file for gollum

# module load python
CHARMMDIR=/home/rhaye/CHARMM7/chv1/charmm_debug
source $CHARMMDIR/modules
CHARMMEXEC=$CHARMMDIR/gnu/charmm

DIR=`pwd`

name=`cat name`
nnodes=`cat nnodes`
nreps=`cat nreps`

ini=1
iri=1
ifi=50

for i in `seq $ini $ifi`
do

im1=$(( $i - 1 ))
ip1=$(( $i + 1 ))
im5=`awk 'BEGIN {if (('$i'-4)<'$iri') {print '$iri'} else {print '$i'-4}}'`
N=$(( $i - $im5 + 1 ))
ir=$(( $RANDOM % ( $ip1 - $iri ) + $iri ))

RUNDIR=$DIR/debug$i
PANADIR=$DIR/analysis$im1
ANADIR=$DIR/analysis$i

while [ ! -f $ANADIR/b_sum.dat ]
do

if [ -d ${RUNDIR}_failed ]; then
  rm -r ${RUNDIR}_failed
fi
if [ -d $RUNDIR ]; then
  cp -r $RUNDIR ${RUNDIR}_failed
  rm -r $RUNDIR
  echo "run$i failed"
  sleep 30
fi

# Run the simulation
mkdir $RUNDIR
mkdir $RUNDIR/res $RUNDIR/dcd
cp -r variables$i.inp prep $RUNDIR/
cd $RUNDIR

# timeout -s SIGINT 8h
echo "run$i started"
# mpirun -np $(($nreps * $nnodes)) -x OMP_NUM_THREADS=4 --bind-to none --bynode $CHARMMEXEC iflat=$i nsteps=50000 seed=$RANDOM -i ../msld_flat.inp > output 2> error
export OMP_NUM_THREADS=1
echo "iflat=$i nsteps=50000 seed=$RANDOM -i ../msld_flat_debug.inp"
cuda-gdb $CHARMMEXEC

exit

cd $DIR


# Run the analysis
echo "analysis$i started"
if [ ! -d $ANADIR ]; then
  mkdir $ANADIR
fi
cp -r analysisPhase2/* $ANADIR/
cp $PANADIR/b_sum.dat $ANADIR/b_prev.dat
cp $PANADIR/c_sum.dat $ANADIR/c_prev.dat
cp $PANADIR/x_sum.dat $ANADIR/x_prev.dat
cp $PANADIR/s_sum.dat $ANADIR/s_prev.dat
cd $ANADIR

./GetLambdas.py $i
./GetEnergy.py $im5 $i
./RunWham.sh $(( $N * $nreps ))
./GetFreeEnergy3.py

./SetVars.py
echo "set restartfile = \"../run$ir/res/${name}_res0_prod.res\"" > $DIR/variables$ip1.inp
cat fb_est.inp vb_est.inp xb_est.inp sb_est.inp parm.inp >> $DIR/variables$ip1.inp

cd $DIR

done

done
