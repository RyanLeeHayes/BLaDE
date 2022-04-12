#! /bin/bash -l

# source modules
module load environments/pgi_2018_10 

mkdir buildPGI
cd buildPGI
cmake -DUNITS=AKMA -DCMAKE_BUILD_TYPE=Release ../src
# cmake -DUNITS=AKMA -DPRECISION=DOUBLE -DCMAKE_BUILD_TYPE=Release ../src

make $1 blade
