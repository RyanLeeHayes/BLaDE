#!/bin/bash
rexrank=1
for i in $(seq -84 17 -50)
	do for j in $(seq -157.5 45 157.5)
		do cat <<EOF >> restraints${rexrank}.inp
! this is the place to put all the restraints
selection define clp atom HETA 1 CLP
selection define nl atom HETA 1 NL
selection define ca atom HETA 1 CA
selection define crp atom HETA 1 CRP
selection define nr atom HETA 1 NR
structure dihedral clp nl ca crp 8 $i 1 1
structure dihedral nl ca crp nr 8 $j 1 1
EOF
rexrank=$((rexrank + 1))
done
done 
