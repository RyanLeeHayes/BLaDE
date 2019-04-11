#! /usr/bin/env python

import os
import copy

ResList=['ALA','CYS','ASP','GLU','PHE','GLY','HSD','HSE','HSP','ILE','LYS','LEU','MET','ASN','PRO','GLN','ARG','SER','THR','VAL','TRP','TYR']
Conv321={'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HSD':'H','HSE':'B','HSP':'J','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}

try:
  os.mkdir('aa_stream')
except:
  pass

fprtf=open('aa_stream/msldpatch.str','w')
fprtf.write("read rtf append card\n* All the patches\n*\n    22     0\n")

ads={} # Atom names
bds={} # Backbone atoms
cds={} # Constrained atoms
for Res in ResList:
  R=Conv321[Res]
  r=R.lower()
  fp=open('top_all36_prot.rtf','r')
  fpsel=open('aa_stream/sele_'+r+'.str','w')
  position=-1
  atomdict={'N':'N'+R+'_A','CA':'C'+R+'_A','C':'C'+R+'_B','O':'O'+R+'_A'}
  backdict={'N':'N'+R+'_A','CA':'C'+R+'_A','C':'C'+R+'_B','O':'O'+R+'_A'}
  consdict={'N':'N'+R+'_A','CA':'C'+R+'_A','C':'C'+R+'_B','O':'O'+R+'_A'}
  if Res=="PRO":
    atomdict['HA']='H'+R+'_B'
    atomdict['CB']='C'+R+'_C'
    atomdict['CG']='C'+R+'_D'
    atomdict['CD']='C'+R+'_E'
    elemcount={'H':2,'C':5,'N':1,'O':1} # pad the hydrogens so there is no HP_A
    backdict['HN']='C'+R+'_E'
    consdict['HA']='H'+R+'_B'
  elif Res=="GLY":
    atomdict['HN']='H'+R+'_A'
    atomdict['HA2']='H'+R+'_B'
    elemcount={'H':2,'C':2,'N':1,'O':1}
    backdict['HN']='H'+R+'_A'
    consdict['HA2']='H'+R+'_B'
  else:
    atomdict['HN']='H'+R+'_A'
    atomdict['HA']='H'+R+'_B'
    elemcount={'H':2,'C':2,'N':1,'O':1}
    backdict['HN']='H'+R+'_A'
    consdict['HA']='H'+R+'_B'
  bondadd="BOND N"+R+"_A -C\n"
  for line in fp:
    if position==-1:
      if "RESI "+Res+" " in line:
        fprtf.write("PRES aa_"+r+line[8:])
        fpsel.write("selection define site{site}sub{sub} and and segid {segid} resid {resid} atomnames")
        position=0
    elif position==0:
      if len(line.split())>0:
        token=line.split()[0]
        if token=="GROUP":
          fprtf.write(line)
        elif token=="ATOM":
          buf=line.split()
          if not buf[1] in atomdict:
            atom=buf[1]
            elem=atom[0]
            if not elem in elemcount:
              elemcount[elem]=0
            elemcount[elem]=elemcount[elem]+1
            count=elemcount[elem]
            atomdict[atom]=(elem+R+'_'+chr(ord('A')+count-1))
          buf[1]=atomdict[buf[1]]
          fprtf.write(" ".join(buf)+"\n")
          fpsel.write(" "+buf[1])
        elif token=="BOND":
          if len(bondadd)>0:
            fprtf.write(bondadd)
            bondadd=""
          buf=line.split()
          for i in range(1,len(buf)):
            if buf[i] in atomdict:
              buf[i]=atomdict[buf[i]]
          fprtf.write(" ".join(buf)+"\n")
        elif token=="DOUBLE":
          buf=line.split()
          for i in range(1,len(buf)):
            if buf[i] in atomdict:
              buf[i]=atomdict[buf[i]]
          fprtf.write(" ".join(buf)+"\n")
        elif token=="IMPR":
          buf=line.split()
          for i in range(1,len(buf)):
            if buf[i] in atomdict:
              buf[i]=atomdict[buf[i]]
          fprtf.write(" ".join(buf)+"\n")
        elif token=="CMAP":
          buf=line.split()
          for i in range(1,len(buf)):
            if buf[i] in atomdict:
              buf[i]=atomdict[buf[i]]
          fprtf.write(" ".join(buf)+"\n")
        elif token=="DONOR":
          buf=line.split()
          for i in range(1,len(buf)):
            if buf[i] in atomdict:
              buf[i]=atomdict[buf[i]]
          fprtf.write(" ".join(buf)+"\n")
        elif token=="ACCEPTOR":
          buf=line.split()
          for i in range(1,len(buf)):
            if buf[i] in atomdict:
              buf[i]=atomdict[buf[i]]
          fprtf.write(" ".join(buf)+"\n")
        elif token=="IC":
          buf=line.split()
          for i in range(1,5):
            if buf[i] in atomdict:
              buf[i]=atomdict[buf[i]]
            elif buf[i][0]=="*" and (buf[i][1:] in atomdict):
              buf[i]="*"+atomdict[buf[i][1:]]
          fprtf.write(" ".join(buf)+"\n")
        elif token=="RESI":
          fprtf.write("\n")
          fpsel.write("\n")
          fpsel.close()
          position=1
        elif token[0]=="!":
          fprtf.write(line)
        elif token=="PATCHING":
          print("Warning: ignoring PATCHING for residue "+Res)
        else:
          print("Unrecognized token: "+token)
          quit()
  fp.close()

  fpsel=open('aa_stream/sele_cons_'+r+'.str','w')
  fpsel.write("selection define site{site}cons{sub} and and segid {segid} resid {resid} atomnames")
  for atom in consdict:
    fpsel.write(" "+consdict[atom])
  fpsel.write("\n")
  fpsel.close()
  
  fpnoe=open('aa_stream/noe_'+r+'.str','w')
  fpnoe.write("* nuissance title\n*\n")
  fpnoe.write("NOE\n")
  fpnoe.write("assign sele atom @segid @resid@@site N"+R+"_A end -\n")
  fpnoe.write("sele segid @segid .and. resid @resid@@site .and. (type N .or. type N*_A) end -\n")
  fpnoe.write("kmin 59.2 rmin 0.0 kmax 59.2 rmax 0.0 fmax 2.0 rswitch 100.0 sexp 1.0\n")
  fpnoe.write("assign sele atom @segid @resid@@site C"+R+"_A end -\n")
  fpnoe.write("sele segid @segid .and. resid @resid@@site .and. (type CA .or. type C*_A) end -\n")
  fpnoe.write("kmin 59.2 rmin 0.0 kmax 59.2 rmax 0.0 fmax 2.0 rswitch 100.0 sexp 1.0\n")
  fpnoe.write("assign sele atom @segid @resid@@site H"+R+"_B end -\n")
  fpnoe.write("sele segid @segid .and. resid @resid@@site .and. (type HA .or. type H*_B) end -\n")
  fpnoe.write("kmin 59.2 rmin 0.0 kmax 59.2 rmax 0.0 fmax 2.0 rswitch 100.0 sexp 1.0\n")
  fpnoe.write("assign sele atom @segid @resid@@site C"+R+"_B end -\n")
  fpnoe.write("sele segid @segid .and. resid @resid@@site .and. (type C .or. type C*_B) end -\n")
  fpnoe.write("kmin 59.2 rmin 0.0 kmax 59.2 rmax 0.0 fmax 2.0 rswitch 100.0 sexp 1.0\n")
  fpnoe.write("assign sele atom @segid @resid@@site O"+R+"_A end -\n")
  fpnoe.write("sele segid @segid .and. resid @resid@@site .and. (type O .or. type O*_A) end -\n")
  fpnoe.write("kmin 59.2 rmin 0.0 kmax 59.2 rmax 0.0 fmax 2.0 rswitch 100.0 sexp 1.0\n")
  fpnoe.write("END\n")
  fpnoe.close()

  ads[Res]=copy.deepcopy(atomdict)
  bds[Res]=copy.deepcopy(backdict)
  cds[Res]=copy.deepcopy(consdict)

fpsel=open('aa_stream/sele_cons_0.str','w')
fpsel.write("selection define site{site}cons1 and and segid {segid} resid {resid} atomnames")
for atom in cds['ALA']:
  fpsel.write(" "+atom)
fpsel.write(" HA2")
fpsel.write("\n")
fpsel.close()

fpnoe=open('aa_stream/noe_0.str','w')
fpnoe.write("* nuissance title\n*\n")
fpnoe.write("NOE\n")
fpnoe.write("assign sele atom @segid @resid@@site N end -\n")
fpnoe.write("sele segid @segid .and. resid @resid@@site .and. (type N .or. type N*_A) end -\n")
fpnoe.write("kmin 59.2 rmin 0.0 kmax 59.2 rmax 0.0 fmax 2.0 rswitch 100.0 sexp 1.0\n")
fpnoe.write("assign sele atom @segid @resid@@site CA end -\n")
fpnoe.write("sele segid @segid .and. resid @resid@@site .and. (type CA .or. type C*_A) end -\n")
fpnoe.write("kmin 59.2 rmin 0.0 kmax 59.2 rmax 0.0 fmax 2.0 rswitch 100.0 sexp 1.0\n")
fpnoe.write("assign sele atom @segid @resid@@site HA .or. atom @segid @resid@@site HA2 end -\n")
fpnoe.write("sele segid @segid .and. resid @resid@@site .and. (type HA .or. type HA2 .or. type H*_B) end -\n")
fpnoe.write("kmin 59.2 rmin 0.0 kmax 59.2 rmax 0.0 fmax 2.0 rswitch 100.0 sexp 1.0\n")
fpnoe.write("assign sele atom @segid @resid@@site C end -\n")
fpnoe.write("sele segid @segid .and. resid @resid@@site .and. (type C .or. type C*_B) end -\n")
fpnoe.write("kmin 59.2 rmin 0.0 kmax 59.2 rmax 0.0 fmax 2.0 rswitch 100.0 sexp 1.0\n")
fpnoe.write("assign sele atom @segid @resid@@site O end -\n")
fpnoe.write("sele segid @segid .and. resid @resid@@site .and. (type O .or. type O*_A) end -\n")
fpnoe.write("kmin 59.2 rmin 0.0 kmax 59.2 rmax 0.0 fmax 2.0 rswitch 100.0 sexp 1.0\n")
fpnoe.write("END\n")
fpnoe.close()

# ------------------------ TERMINAL PATCHES ------------------------------------
N321={'NAT':'0','NATP':'1','NTER':'2','CTER':'3','ACE':'4','CT3':'5'}

for Res in ResList:
  R=Conv321[Res]
  r=R.lower()
  for Cap in ['NTER','CTER','ACE','CT3']:
    n=N321[Cap]
    Pres=Cap
    atomdict=copy.deepcopy(ads[Res])
    if Cap=='NTER':
      if Res=='PRO':
        atomdict.update({'HN1':'H'+R+'TA','HN2':'H'+R+'TB'})
      else:
        atomdict.update({'HT1':'H'+R+'TA','HT2':'H'+R+'TB','HT3':'H'+R+'TC'})
    elif Cap=='CTER':
      atomdict.update({'OT1':'O'+R+'TA','OT2':'O'+R+'TB'})
    elif Cap=='ACE':
      atomdict.update({'HY1':'H'+R+'TA','HY2':'H'+R+'TB','HY3':'H'+R+'TC','CAY':'C'+R+'TA','CY':'C'+R+'TB','OY':'O'+R+'TA'})
    elif Cap=='CT3':
      atomdict.update({'NT':'N'+R+'TA','HNT':'H'+R+'TA','CAT':'C'+R+'TA','HT1':'H'+R+'TB','HT2':'H'+R+'TC','HT3':'H'+R+'TD'})
    if Res=='GLY':
      if Cap=='NTER':
        Pres='GLP'
    elif Res=='PRO':
      if Cap=='NTER':
        Pres='PROP'
      elif Cap=='ACE':
        Pres='ACP'

    fp=open('top_all36_prot.rtf','r')
    fpsel=open('aa_stream/sele_'+r+n+'.str','w')
    position=-1
    for line in fp:
      if position==-1:
        if "PRES "+Pres+" " in line:
          fprtf.write("PRES cap_"+r+n+" "+" ".join(line.split()[2:])+"\n")
          fpsel.write("selection define site{site}cap{sub} and and segid {segid} resid {resid} atomnames")
          position=0
      elif position==0:
        if len(line.split())>0:
          token=line.split()[0]
          if token=="GROUP":
            fprtf.write(line)
          elif token=="ATOM":
            buf=line.split()
            if not buf[1] in atomdict:
              print("Error: atom "+buf[1]+" must be in atomdict for capping.\n")
              print("Residue="+Res+" Cap="+Cap)
              print(atomdict)
              quit()
              atom=buf[1]
              elem=atom[0]
              if not elem in elemcount:
                elemcount[elem]=0
              elemcount[elem]=elemcount[elem]+1
              count=elemcount[elem]
              atomdict[atom]=(elem+R+'_'+chr(ord('A')+count-1))
            buf[1]=atomdict[buf[1]]
            fprtf.write(" ".join(buf)+"\n")
            fpsel.write(" "+buf[1])
          elif token=="DELETE":
            buf=line.split()
            for i in range(1,len(buf)):
              if buf[i] in atomdict:
                buf[i]=atomdict[buf[i]]
            fprtf.write(" ".join(buf)+"\n")
          elif token=="BOND":
            buf=line.split()
            for i in range(1,len(buf)):
              if buf[i] in atomdict:
                buf[i]=atomdict[buf[i]]
            fprtf.write(" ".join(buf)+"\n")
          elif token=="DOUBLE":
            buf=line.split()
            for i in range(1,len(buf)):
              if buf[i] in atomdict:
                buf[i]=atomdict[buf[i]]
            fprtf.write(" ".join(buf)+"\n")
          elif token=="IMPR":
            buf=line.split()
            for i in range(1,len(buf)):
              if buf[i] in atomdict:
                buf[i]=atomdict[buf[i]]
            fprtf.write(" ".join(buf)+"\n")
          elif token=="CMAP":
            buf=line.split()
            for i in range(1,len(buf)):
              if buf[i] in atomdict:
                buf[i]=atomdict[buf[i]]
            fprtf.write(" ".join(buf)+"\n")
          elif token=="DONOR":
            buf=line.split()
            for i in range(1,len(buf)):
              if buf[i] in atomdict:
                buf[i]=atomdict[buf[i]]
            fprtf.write(" ".join(buf)+"\n")
          elif token=="ACCEPTOR":
            buf=line.split()
            for i in range(1,len(buf)):
              if buf[i] in atomdict:
                buf[i]=atomdict[buf[i]]
            fprtf.write(" ".join(buf)+"\n")
          elif token=="IC":
            buf=line.split()
            for i in range(1,5):
              if buf[i] in atomdict:
                buf[i]=atomdict[buf[i]]
              elif buf[i][0]=="*" and (buf[i][1:] in atomdict):
                buf[i]="*"+atomdict[buf[i][1:]]
            fprtf.write(" ".join(buf)+"\n")
          elif token=="RESI" or token=="PRES":
            fprtf.write("\n")
            fpsel.write("\n")
            position=1
          elif token[0]=="!":
            fprtf.write(line)
          elif token=="PATCHING":
            print("Warning: ignoring PATCHING for residue "+Res)
          else:
            print("Unrecognized token: "+token)
            quit()
    fp.close()

# ---------------------------------- LINK PATCHES ------------------------------

BOND1="BOND -C N"
BOND2="BOND C +N"
IMPR1="IMPR N -C CA HN"
IMPR2="IMPR C CA +N O"
CMAP="CMAP -C  N  CA  C   N  CA  C  +N"

def topsub(t,d1,d2,d3):
  ts=t.split()
  for i in range(1,len(ts)):
    atom=ts[i]
    if atom[0]=="+":
      atom="+"+d3[atom[1:]]
    elif atom[0]=="-":
      atom="-"+d1[atom[1:]]
    else:
      atom=d2[atom]
    ts[i]=atom
  return " ".join(ts)+"\n"

def topsubn(t,d2,d3):
  ts=t.split()
  for i in range(1,len(ts)):
    atom=ts[i]
    if atom[0]=="+":
      atom="+"+d3[atom[1:]]
    else:
      atom=d2[atom]
    ts[i]=atom
  return " ".join(ts)+"\n"

def topsubc(t,d1,d2):
  ts=t.split()
  for i in range(1,len(ts)):
    atom=ts[i]
    if atom[0]=="-":
      atom="-"+d1[atom[1:]]
    else:
      atom=d2[atom]
    ts[i]=atom
  return " ".join(ts)+"\n"

atomdicts={}
atomdictnts={}
atomdictcts={}
atomdictncs={}
atomdictccs={}
for Res in ResList:
  R=Conv321[Res]
  atomdicts[Res]={'N':'N'+R+'_A','HN':'H'+R+'_A','CA':'C'+R+'_A','C':'C'+R+'_B','O':'O'+R+'_A'}
  atomdictnts[Res]={'N':'N'+R+'_A','CA':'C'+R+'_A','C':'C'+R+'_B','O':'O'+R+'_A'}
  atomdictcts[Res]={'N':'N'+R+'_A','HN':'H'+R+'_A','CA':'C'+R+'_A','C':'C'+R+'_B'}
  atomdictncs[Res]={'N':'N'+R+'_A','HN':'H'+R+'_A','CA':'C'+R+'_A','C':'C'+R+'_B','O':'O'+R+'_A','-CA':'C'+R+'TA','-C':'C'+R+'TB','-O':'O'+R+'TA'}
  atomdictccs[Res]={'N':'N'+R+'_A','HN':'H'+R+'_A','CA':'C'+R+'_A','C':'C'+R+'_B','O':'O'+R+'_A','+N':'N'+R+'TA','+HN':'H'+R+'TA','+CA':'C'+R+'TA'}
  if R=="P":
    # HN is actually CD, or CP_E
    atomdicts[Res]['HN']='C'+R+'_E'
    # No 'HN' in atomdictnts
    atomdictcts[Res]['HN']='C'+R+'_E'
    atomdictncs[Res]['HN']='C'+R+'_E'
    atomdictccs[Res]['HN']='C'+R+'_E'
atomdict_n={'N':'N','HN':'HN','CA':'CA','C':'C','O':'O'}
atomdictnt_n={'N':'N','CA':'CA','C':'C','O':'O'}
atomdictct_n={'N':'N','HN':'HN','CA':'CA','C':'C'}
atomdictnc_n={'N':'N','HN':'HN','CA':'CA','C':'C','O':'O','-CA':'CAY','-C':'CY','-O':'OY'}
atomdictcc_n={'N':'N','HN':'HN','CA':'CA','C':'C','O':'O','+N':'NT','+HN':'HNT','+CA':'CAT'}
atomdict_np=copy.deepcopy(atomdict_n)
atomdictnt_np=copy.deepcopy(atomdictnt_n)
atomdictct_np=copy.deepcopy(atomdictct_n)
atomdictnc_np=copy.deepcopy(atomdictnc_n)
atomdictcc_np=copy.deepcopy(atomdictcc_n)
atomdict_np['HN']='CD'
# atomdictnt_np['HN']='CD'
atomdictct_np['HN']='CD'
atomdictnc_np['HN']='CD'
atomdictcc_np['HN']='CD'

# A_x_x
for Res1 in ResList: # Normal
  r1=Conv321[Res1].lower()
  name="l_"+Conv321[Res1].lower()+N321['NAT']+N321['NAT']
  fprtf.write("PRES "+name+"         0.00\n")
  fprtf.write(topsub(IMPR1,atomdicts[Res1],atomdict_n,atomdict_n))
  fprtf.write(topsub(CMAP,atomdicts[Res1],atomdict_n,atomdict_n))
  fprtf.write("\n")
for Res1 in ResList: # Neighboring proline
  name="l_"+Conv321[Res1].lower()+N321['NATP']+N321['NAT']
  fprtf.write("PRES "+name+"         0.00\n")
  fprtf.write(topsub(IMPR1,atomdicts[Res1],atomdict_np,atomdict_n))
  fprtf.write(topsub(CMAP,atomdicts[Res1],atomdict_np,atomdict_n))
  fprtf.write("\n")
for Res1 in ResList: # Normal (CTER cap)
  name="l_"+Conv321[Res1].lower()+N321['NAT']+N321['CTER']
  fprtf.write("PRES "+name+"         0.00\n")
  fprtf.write(topsubc(IMPR1,atomdicts[Res1],atomdictct_n))
  fprtf.write("\n")
for Res1 in ResList: # Neighboring proline (CTER cap)
  name="l_"+Conv321[Res1].lower()+N321['NATP']+N321['CTER']
  fprtf.write("PRES "+name+"         0.00\n")
  fprtf.write(topsubc(IMPR1,atomdicts[Res1],atomdictct_np))
  fprtf.write("\n")
for Res1 in ResList: # Normal (CT3 cap)
  name="l_"+Conv321[Res1].lower()+N321['NAT']+N321['CT3']
  fprtf.write("PRES "+name+"         0.00\n")
  fprtf.write(topsubc(IMPR1,atomdicts[Res1],atomdictcc_n))
  fprtf.write(topsubc(CMAP,atomdicts[Res1],atomdictcc_n))
  fprtf.write("\n")
for Res1 in ResList: # Neighboring proline (CT3 cap)
  name="l_"+Conv321[Res1].lower()+N321['NATP']+N321['CT3']
  fprtf.write("PRES "+name+"         0.00\n")
  fprtf.write(topsubc(IMPR1,atomdicts[Res1],atomdictcc_np))
  fprtf.write(topsubc(CMAP,atomdicts[Res1],atomdictcc_np))
  fprtf.write("\n")

# x_x_A
for Res3 in ResList: # Normal
  name="l_"+N321['NAT']+N321['NAT']+Conv321[Res3].lower()
  fprtf.write("PRES "+name+"         0.00\n")
  fprtf.write(topsub(IMPR2,atomdict_n,atomdict_n,atomdicts[Res3]))
  fprtf.write(topsub(CMAP,atomdict_n,atomdict_n,atomdicts[Res3]))
  fprtf.write("\n")
for Res3 in ResList: # Normal (NTER)
  name="l_"+N321['NTER']+N321['NAT']+Conv321[Res3].lower()
  fprtf.write("PRES "+name+"         0.00\n")
  fprtf.write(topsubn(IMPR2,atomdictnt_n,atomdicts[Res3]))
  fprtf.write("\n")
for Res3 in ResList: # Normal (ACE cap)
  name="l_"+N321['ACE']+N321['NAT']+Conv321[Res3].lower()
  fprtf.write("PRES "+name+"         0.00\n")
  fprtf.write(topsubn(IMPR2,atomdictnc_n,atomdicts[Res3]))
  fprtf.write(topsubn(CMAP,atomdictnc_n,atomdicts[Res3]))
  fprtf.write("\n")

# A_A_x
for Res1 in ResList: # Normal
  for Res2 in ResList:
    name="l_"+Conv321[Res1].lower()+Conv321[Res2].lower()+N321['NAT']
    fprtf.write("PRES "+name+"         0.00\n")
    fprtf.write(topsub(BOND1,atomdicts[Res1],atomdicts[Res2],atomdict_n))
    fprtf.write(topsub(IMPR1,atomdicts[Res1],atomdicts[Res2],atomdict_n))
    fprtf.write(topsub(CMAP,atomdicts[Res1],atomdicts[Res2],atomdict_n))
    fprtf.write("\n")
for Res1 in ResList: # Normal (CTER)
  for Res2 in ResList:
    name="l_"+Conv321[Res1].lower()+Conv321[Res2].lower()+N321['CTER']
    fprtf.write("PRES "+name+"         0.00\n")
    fprtf.write(topsubc(BOND1,atomdicts[Res1],atomdictcts[Res2]))
    fprtf.write(topsubc(IMPR1,atomdicts[Res1],atomdictcts[Res2]))
    fprtf.write("\n")
for Res1 in ResList: # Normal (CT3)
  for Res2 in ResList:
    name="l_"+Conv321[Res1].lower()+Conv321[Res2].lower()+N321['CT3']
    fprtf.write("PRES "+name+"         0.00\n")
    fprtf.write(topsubc(BOND1,atomdicts[Res1],atomdictccs[Res2]))
    fprtf.write(topsubc(IMPR1,atomdicts[Res1],atomdictccs[Res2]))
    fprtf.write(topsubc(CMAP,atomdicts[Res1],atomdictccs[Res2]))
    fprtf.write("\n")

# A_x_A
for Res1 in ResList:
  for Res3 in ResList:
    name="l_"+Conv321[Res1].lower()+N321['NAT']+Conv321[Res3].lower()
    fprtf.write("PRES "+name+"         0.00\n")
    fprtf.write(topsub(CMAP,atomdicts[Res1],atomdict_n,atomdicts[Res3]))
    fprtf.write("\n")

# x_A_A
for Res2 in ResList: # Normal
  for Res3 in ResList:
    name="l_"+N321['NAT']+Conv321[Res2].lower()+Conv321[Res3].lower()
    fprtf.write("PRES "+name+"         0.00\n")
    fprtf.write(topsub(IMPR2,atomdict_n,atomdicts[Res2],atomdicts[Res3]))
    fprtf.write(topsub(CMAP,atomdict_n,atomdicts[Res2],atomdicts[Res3]))
    fprtf.write("\n")
for Res2 in ResList: # Normal (NTER)
  for Res3 in ResList:
    name="l_"+N321['NTER']+Conv321[Res2].lower()+Conv321[Res3].lower()
    fprtf.write("PRES "+name+"         0.00\n")
    fprtf.write(topsubn(IMPR2,atomdictnts[Res2],atomdicts[Res3]))
    fprtf.write("\n")
for Res2 in ResList: # Normal (ACE)
  for Res3 in ResList:
    name="l_"+N321['ACE']+Conv321[Res2].lower()+Conv321[Res3].lower()
    fprtf.write("PRES "+name+"         0.00\n")
    fprtf.write(topsubn(IMPR2,atomdictncs[Res2],atomdicts[Res3]))
    fprtf.write(topsubn(CMAP,atomdictncs[Res2],atomdicts[Res3]))
    fprtf.write("\n")

# A_A_A
for Res1 in ResList:
  for Res2 in ResList:
    for Res3 in ResList:
      name="l_"+Conv321[Res1].lower()+Conv321[Res2].lower()+Conv321[Res3].lower()
      fprtf.write("PRES "+name+"         0.00\n")
      fprtf.write(topsub(CMAP,atomdicts[Res1],atomdicts[Res2],atomdicts[Res3]))
      fprtf.write("\n")

fprtf.write("end\nreturn\n\n")
fprtf.close()
