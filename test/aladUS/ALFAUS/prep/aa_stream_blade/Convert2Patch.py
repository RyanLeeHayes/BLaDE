#! /usr/bin/env python

import os
import copy

ResList=['ALA','CYS','ASP','GLU','PHE','GLY','HSD','HSE','HSP','ILE','LYS','LEU','MET','ASN','PRO','GLN','ARG','SER','THR','VAL','TRP','TYR']
Conv321={'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HSD':'H','HSE':'B','HSP':'J','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}

fprtf=open('msldpatch.str','w')
fprtf.write("read rtf append card\n* All the patches\n*\n    22     0\n")

ads={} # Atom names
for Res in ResList:
  R=Conv321[Res]
  r=R.lower()
  fp=open('top_all36_prot.rtf','r')
  position=-1
  atomdict={'N':'N__'+R,'C':'C__'+R}
  if Res=="ILE":                #Corrects my pet peeve of Ile's 'HG11' becoming 'HG4I' while 'HG21' becomes 'HG1I'
    atomdict['HG11']='HG1'+R
    atomdict['HG12']='HG2'+R
    atomdict['HG21']='HG3'+R
    atomdict['HG22']='HG4'+R
    atomdict['HG23']='HG5'+R
  twocharcount={}
  bondadd="BOND N__"+R+" -C\n"
  for line in fp:
    if position==-1:
      if "RESI "+Res+" " in line:
        fprtf.write("PRES aa_"+r+line[9:])
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
            if len(atom)==1:
              newatom=atom+'__'+R
            elif len(atom)==2:
              newatom=atom+'_'+R
            elif len(atom)==3:
              newatom=atom+R
              twochar = atom[0:2]             #Without these 4 lines: Thr's 'HG1' and 'HG21' both become 'HG1T'
              if not twochar in twocharcount:
                twocharcount[twochar]=0
              twocharcount[twochar]=twocharcount[twochar]+1
            else:
              twochar=atom[0:2]
              if not twochar in twocharcount:
                twocharcount[twochar]=0
              twocharcount[twochar]=twocharcount[twochar]+1
              newatom=twochar+str(twocharcount[twochar])+R
            atomdict[atom]=newatom
          newline = line.replace((buf[1]+(' '*(len(atomdict[buf[1]])-len(buf[1])))),atomdict[buf[1]],1)
          fprtf.write(newline)
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
          newline = ''
          for i in range(5):
            if i in [1,2,4]:
              newline += buf[i]+(' '*(5-len(buf[i])))
            elif i == 3:
              newline += buf[i]+(' '*(6-len(buf[i])))
            else:
              newline += buf[i]+' '
          newline += line[24:]
          fprtf.write(newline)
        elif token=="RESI":
          fprtf.write("\n")
          position=1
        elif token[0]=="!":
          fprtf.write(line)
        elif token=="PATCHING":
          print("Warning: ignoring PATCHING for residue "+Res)
        else:
          print("Unrecognized token: "+token)
          quit()
  fp.close()

  ads[Res]=copy.deepcopy(atomdict)

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
        capatoms=['HN1','HN2']
      else:
        capatoms=['HT1','HT2','HT3']
    elif Cap=='CTER':
      capatoms=['OT1','OT2']
    elif Cap=='ACE':
      capatoms=['HY1','HY2','HY3','CAY','CY','OY']
    elif Cap=='CT3':
      capatoms=['NT','HNT','CAT','HT1','HT2','HT3']
    if Res=='GLY':
      if Cap=='NTER':
        Pres='GLYP'
    elif Res=='PRO':
      if Cap=='NTER':
        Pres='PROP'
      elif Cap=='ACE':
        Pres='ACP'

    fp=open('top_all36_prot.rtf','r')
    position=-1
    for line in fp:
      if position==-1:
        if "PRES "+Pres+" " in line:
          if '-' in line[:23]:
            newline = ("PRES cap_"+r+n+"      "+" ".join(line.split()[2:])+"\n")
          else:
            newline = ("PRES cap_"+r+n+"       "+" ".join(line.split()[2:])+"\n")
          if r+n == 'p2':
            newline = newline.replace('!','  !',1)
          fprtf.write(newline)
          position=0
      elif position==0:
        if len(line.split())>0:
          token=line.split()[0]
          if token=="GROUP":
            if r+n == 'p2':
              line = line.replace('!','  !',1)
            fprtf.write(line)
          elif token=="ATOM":
            buf=line.split()
            if not buf[1] in atomdict:
              if not buf[1] in capatoms:
                print("Error: atom "+buf[1]+" must be in atomdict or cap atoms for capping.\n")
                print("Residue="+Res+" Cap="+Cap)
                print(atomdict)
                print(capatoms)
                quit()
              atom=buf[1]
              if len(atom)==1:
                newatom=atom+'__'+R
              elif len(atom)==2:
                newatom=atom+'_'+R
              elif len(atom)==3:
                newatom=atom+R
              else:
                twochar=atom[0:2]
                if not twochar in twocharcount:
                  twocharcount[twochar]=0
                twocharcount[twochar]=twocharcount[twochar]+1
                newatom=twochar+str(twocharcount[twochar])+R
              atomdict[atom]=newatom
            newline = line.replace((buf[1]+(' '*(len(atomdict[buf[1]])-len(buf[1])))),atomdict[buf[1]],1)
            if r+n == 'p2':
              newline = newline.replace('!','  !',1)
            fprtf.write(newline)
          elif token=="DELETE":
            buf=line.split()
            for i in range(1,len(buf)):
              if buf[i] in atomdict:
                buf[i]=atomdict[buf[i]]
            if '!' in buf:
              cut = buf.index('!')
              newline = " ".join(buf[:cut])
              newline += (' '*(23 - len(newline)))
              newline += line[23:]
              fprtf.write(newline)
            else:
              fprtf.write(" ".join(buf)+"\n")
          elif token=="BOND":
            buf=line.split()
            for i in range(1,len(buf)):
              if buf[i] in atomdict:
                buf[i]=atomdict[buf[i]]
            if '!' in buf:
              cut = buf.index('!')
              newline = " ".join(buf[:cut])
              newline += (' '*(23 - len(newline)))
              newline += line[23:]
            else:
              newline = (" ".join(buf)+"\n")
            if r+n == 'p2':
              newline = newline.replace('!',' !',1)
            fprtf.write(newline)
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
            newline = ''
            for i in range(5):
              if i in [1,2,4]:
                newline += buf[i]+(' '*(5-len(buf[i])))
              elif i == 3:
                newline += buf[i]+(' '*(6-len(buf[i])))
              else:
                newline += buf[i]+' '
            newline += line[23:]
            newline = newline.replace('0.0000 \n','0.0000\n',1)
            fprtf.write(newline)
          elif token=="RESI" or token=="PRES":
            fprtf.write("\n")
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
  # atomdicts[Res]={'N':'N'+R+'_A','HN':'H'+R+'_A','CA':'C'+R+'_A','C':'C'+R+'_B','O':'O'+R+'_A'}
  # atomdictnts[Res]={'N':'N'+R+'_A','CA':'C'+R+'_A','C':'C'+R+'_B','O':'O'+R+'_A'}
  # atomdictcts[Res]={'N':'N'+R+'_A','HN':'H'+R+'_A','CA':'C'+R+'_A','C':'C'+R+'_B'}
  # atomdictncs[Res]={'N':'N'+R+'_A','HN':'H'+R+'_A','CA':'C'+R+'_A','C':'C'+R+'_B','O':'O'+R+'_A','-CA':'C'+R+'TA','-C':'C'+R+'TB','-O':'O'+R+'TA'}
  # atomdictccs[Res]={'N':'N'+R+'_A','HN':'H'+R+'_A','CA':'C'+R+'_A','C':'C'+R+'_B','O':'O'+R+'_A','+N':'N'+R+'TA','+HN':'H'+R+'TA','+CA':'C'+R+'TA'}
  atomdicts[Res]=copy.deepcopy(ads[Res])
  atomdictnts[Res]=copy.deepcopy(ads[Res])
  atomdictcts[Res]=copy.deepcopy(ads[Res])
  atomdictncs[Res]=copy.deepcopy(ads[Res])
  atomdictccs[Res]=copy.deepcopy(ads[Res])
  atomdictncs[Res].update({'-CA':'CAY'+R,'-C':'CY_'+R,'-O':'OY_'+R})
  atomdictccs[Res].update({'+N':'NT_'+R,'+HN':'HNT','+CA':'CAT'+R})
  if R=="P":
    # HN is actually CD
    atomdicts[Res]['HN']='CD_'+R
    # No 'HN' in atomdictnts
    atomdictcts[Res]['HN']='CD_'+R
    atomdictncs[Res]['HN']='CD_'+R
    atomdictccs[Res]['HN']='CD_'+R
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

fprtf.write("end\nreturn\n")
fprtf.close()
