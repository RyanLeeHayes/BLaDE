#!/bin/bash

if [ -z "$BLADEDIR" ]; then
  echo "Error, forgot to set BLADEDIR"
  exit
fi
if [ -z "$BLADEEXE" ]; then
  echo "Error, forgot to set BLADEEXE"
  exit
fi

DIR=`pwd`
export OMP_NUM_THREADS=1

echo T4L
cp -r $BLADEDIR/test/T4L ./
cd T4L
mkdir run112a
mkdir run112a/res run112a/dcd
ln -s `pwd`/prep `pwd`/run112a/prep
cp variables112.inp run112a/variablesprod.inp
cp arguments.inp run112a/
cd run112a
time $BLADEEXE ../msld_prod.inp > output 2> error
cd $DIR

echo RNaseH
cp -r $BLADEDIR/test/RNaseH ./
cd RNaseH
mkdir run37a
mkdir run37a/res run37a/dcd
ln -s `pwd`/prep `pwd`/run37a/prep
cp arguments.inp run37a/
cp variables37.inp run37a/variablesprod.inp
cd run37a
time $BLADEEXE ../msld_prod.inp > output 2> error
cd $DIR

echo HSP90
cp -r $BLADEDIR/test/HSP90 ./
cd HSP90
mkdir run112a
mkdir run112a/res run112a/dcd
ln -s `pwd`/prep `pwd`/run112a/prep
cp arguments.inp run112a/
cp variables112.inp run112a/variablesprod.inp
cd run112a
time $BLADEEXE ../msld_prod.inp > output 2> error
cd $DIR

echo newDHFR
cp -r $BLADEDIR/test/newDHFR ./
cd newDHFR
time $BLADEEXE m.inp > output 2> error
cd $DIR

echo dmpg
cp -r $BLADEDIR/test/dmpg ./
cd dmpg
time $BLADEEXE dmpg290k_new.inp > output 2> error
cd $DIR
