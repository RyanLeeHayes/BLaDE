
for p in ala arg asn asp cys gln glu hsd hse hsp ile leu lys met phe ser thr trp tyr val
do

echo "* nuissance title" > sele_$p.str
echo "*" >> sele_$p.str

echo "define site@{resid}sub$p -" >> sele_$p.str
echo "   select ( -" >> sele_$p.str
echo "   segid @segid .and. resid @resid .and. ( -" >> sele_$p.str

for a in `awk '{if ($1=="ATOM") {print $2}}' ca_$p.str`
do
 
echo "   type $a .or. -" >> sele_$p.str

done

echo "   none ) ) end" >> sele_$p.str

done
