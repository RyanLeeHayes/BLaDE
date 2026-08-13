#! /usr/bin/env python

## (\M/) Monica Barron 7/20/2023
## Modify Ryan's proline/glycine files to have atom names to have preferred format:
##
## atom name = atom type + position on amino acid sidechain + distinguishing number (as needed) + amino acid 1 letter ID
## all atom name's are set to be four characters long (just like Ryan's), with '_' as a placeholder in the second/third position
## i.e.: HG3V is Valine's third gamma hydrogen, CB_V is Valine's solitary beta carbon, and N__V is Valine's backbone nitrogen
## 
## Ryan's atom name format is: atom type + amino acid 1 letter ID + '_' + alphabetical counter
## i.e.: HV_F is the sixth hydrogen on Valine, CV_C is the third carbon on Valine, and NV_A is the first nitrogen on Valine
## Corresponding to Valine's third gamma hydrogen, Valine's solitary beta carbon, and Valine's backbone nitrogen, respectively

import os
import sys

readfile1 = './top_all36_prot.rtf'
writefile1 = 'msldpatch.str'
op1 = open(writefile1,'w')

op1.write('read rtf append card\n')
op1.write('* All the patches\n')
op1.write('*\n')
op1.write('    22     0\n')

#alphabit = 'abcdefghijklmnpqrstvwy' #excludes ['o','u','x','z'] which are not in the normal amino acids
alphabit = 'acdefghbjiklmnpqrstvwy' #same as above, but out of aphabetical order to put hsd/hse/hsp togeter: 'hbj'
active = 0
mut_ids = {'a':'ala','b':'hse','c':'cys','d':'asp','e':'glu','f':'phe','g':'gly','h':'hsd','i':'ile','j':'hsp','k':'lys','l':'leu','m':'met','n':'asn','p':'pro','q':'gln','r':'arg','s':'ser','t':'thr','v':'val','w':'trp','y':'tyr'}
char4_replacements = {'arg_HH11':'HH1R','arg_HH12':'HH2R','arg_HH21':'HH3R','arg_HH22':'HH4R','asn_HD21':'HD1N','asn_HD22':'HD2N','gln_HE21':'HE1Q','gln_HE22':'HE2Q','ile_HG11':'HG1I','ile_HG12':'HG2I','ile_HG21':'HG3I','ile_HG22':'HG4I','ile_HG23':'HG5I','leu_HD11':'HD1L','leu_HD12':'HD2L','leu_HD13':'HD3L','leu_HD21':'HD4L','leu_HD22':'HD5L','leu_HD23':'HD6L','thr_HG21':'HG2T','thr_HG22':'HG3T','thr_HG23':'HG4T','val_HG11':'HG1V','val_HG12':'HG2V','val_HG13':'HG3V','val_HG21':'HG4V','val_HG22':'HG5V','val_HG23':'HG6V'}
for char in alphabit:
    replacements = {}
    bondline = 0
    for line in open(readfile1,'r'):
        if (line[:5] == 'RESI ') and (line[5:9] == ((mut_ids[char].upper()) + ' ')):
            active = 1
            line = line.replace('RESI', 'PRES')
            line = line.replace((mut_ids[char].upper()+' '), 'aa_'+char)
            op1.write(line)
        elif active:
            if (line[:5] == 'RESI ') and ((line[5:9] != (mut_ids[char].upper()) + ' ')):
                active = 0
            elif line[:5] == 'ATOM ':
                atname = line.split()[1]
                if len(atname) < 4:
                    toreplace = atname+(' '*(4-len(atname)))
                    replacewith = atname+('_'*(3-len(atname)))+char.upper()
                if len(atname) == 4:
                    toreplace = atname
                    replacewith = char4_replacements[mut_ids[char]+'_'+atname]
                if atname not in replacements:
                    replacements[atname] = replacewith
                line = line.replace(toreplace, replacewith, 1)
                op1.write(line)
            elif line[:3] == 'IC ':
                tmp = line.split()
                for i in range(len(tmp)):
                    atname = tmp[i]
                    if (atname[0] == '*') and (atname[1:] in replacements):
                        tmp[i] = '*'+replacements[atname[1:]]
                    if atname in replacements:
                        tmp[i] = replacements[atname]
                new_line = ''
                for i in range(5):
                    if i in [1,2,4]:
                        new_line += tmp[i]+(' '*(5-len(tmp[i])))
                    elif i == 3:
                        new_line += tmp[i]+(' '*(6-len(tmp[i])))
                    else:
                        new_line += tmp[i]+' '
                new_line += line[24:]
                op1.write(new_line)
            elif (line[:5] == 'BOND ') or (line[:5] == 'IMPR ') or (line[:5] == 'CMAP ') or (line[:7] == 'DOUBLE ') or (line[:6] == 'DONOR ') or (line[:9] == 'ACCEPTOR '):
                if line[:5] == 'BOND ':
                    if bondline == 0:
                        op1.write('BOND N__'+char.upper()+' -C\n')
                    bondline += 1
                tmp = line.split()
                for i in range(len(tmp)):
                    atname = tmp[i]
                    if atname in replacements:
                        tmp[i] = replacements[atname]
                new_line = ''
                for i in range(len(tmp)):
                    new_line += tmp[i]+' '
                new_line = new_line[:-1] + '\n'
                op1.write(new_line)
            else:
                if line[:8] != 'PATCHING':
                    op1.write(line)

for char in alphabit:
    charUC = char.upper()
    ##The three if/elif statements below create NTER/PROP/GLYP patches for all residues
    if char not in 'pg':
        op1.write('PRES cap_'+char+'2       1.00 ! standard N-terminus\n')
        op1.write('GROUP                  ! use in generate statement\n')
        op1.write('ATOM N__'+charUC+' NH3    -0.30 !\n')
        op1.write('ATOM HT1'+charUC+' HC      0.33 !         HT1\n')
        op1.write('ATOM HT2'+charUC+' HC      0.33 !     (+)/\n')
        op1.write('ATOM HT3'+charUC+' HC      0.33 ! --CA--N--HT2\n')
        op1.write('ATOM CA_'+charUC+' CT1     0.21 !   |    \\\n')
        op1.write('ATOM HA_'+charUC+' HB1     0.10 !   HA    HT3\n')
        op1.write('DELETE ATOM HN_'+charUC+'\n')
        op1.write('BOND HT1'+charUC+' N__'+charUC+' HT2'+charUC+' N__'+charUC+' HT3'+charUC+' N__'+charUC+'\n')
        op1.write('DONOR HT1'+charUC+' N__'+charUC+'\n')
        op1.write('DONOR HT2'+charUC+' N__'+charUC+'\n')
        op1.write('DONOR HT3'+charUC+' N__'+charUC+'\n')
        op1.write('IC HT1'+charUC+' N__'+charUC+' CA_'+charUC+'  C__'+charUC+'  0.0000  0.0000  180.0000  0.0000  0.0000\n')
        op1.write('IC HT2'+charUC+' CA_'+charUC+' *N__'+charUC+' HT1'+charUC+'  0.0000  0.0000  120.0000  0.0000  0.0000\n')
        op1.write('IC HT3'+charUC+' CA_'+charUC+' *N__'+charUC+' HT2'+charUC+'  0.0000  0.0000  120.0000  0.0000  0.0000\n')
        op1.write('\n')

    elif char == 'g':
        op1.write('PRES cap_g2       1.00 ! Glycine N-terminus\n')
        op1.write('GROUP                  ! use in generate statement\n')
        op1.write('ATOM N__G NH3    -0.30 !\n')
        op1.write('ATOM HT1G HC      0.33 !   HA1   HT1\n')
        op1.write('ATOM HT2G HC      0.33 !   | (+)/\n')
        op1.write('ATOM HT3G HC      0.33 ! --CA--N--HT2\n')
        op1.write('ATOM CA_G CT2     0.13 !   |    \\\n')
        op1.write('ATOM HA1G HB2     0.09 !   HA2   HT3\n')
        op1.write('ATOM HA2G HB2     0.09 !\n')
        op1.write('DELETE ATOM HN_G\n')
        op1.write('BOND HT1G N__G HT2G N__G HT3G N__G\n')
        op1.write('DONOR HT1G N__G\n')
        op1.write('DONOR HT2G N__G\n')
        op1.write('DONOR HT3G N__G\n')
        op1.write('IC HT1G N__G CA_G  C__G  0.0000  0.0000  180.0000  0.0000  0.0000\n')
        op1.write('IC HT2G CA_G *N__G HT1G  0.0000  0.0000  120.0000  0.0000  0.0000\n')
        op1.write('IC HT3G CA_G *N__G HT2G  0.0000  0.0000  120.0000  0.0000  0.0000\n')
        op1.write('\n')

    elif char == 'p':
        op1.write('PRES cap_p2       1.00   ! Proline N-Terminal\n')
        op1.write('GROUP                    ! use in generate statement\n')
        op1.write('ATOM N__P NP     -0.07   !   HA\n')
        op1.write('ATOM HN1P HC      0.24   !   |\n')
        op1.write('ATOM HN2P HC      0.24   !  -CA   HN1\n')
        op1.write('ATOM CD_P CP3     0.16   !  /  \ /\n')
        op1.write('ATOM HD1P HA2     0.09   !       N(+)\n')
        op1.write('ATOM HD2P HA2     0.09   !      / \\\n')
        op1.write('ATOM CA_P CP1     0.16   !  -CD    HN2\n')
        op1.write('ATOM HA_P HB1     0.09   !   | \\\n')
        op1.write('BOND HN1P N__P HN2P N__P !  HD1 HD2\n')
        op1.write('DONOR HN1P N__P\n')
        op1.write('DONOR HN2P N__P\n')
        op1.write('IC HN1P CA_P *N__P CD_P  0.0000  0.0000  120.0000  0.0000  0.0000\n')
        op1.write('IC HN2P CA_P *N__P HN1P  0.0000  0.0000  120.0000  0.0000  0.0000\n')
        op1.write('\n')

#for char in alphabit:
#    charUC = char.upper()
    ##The (uncessary but aestetichally pleasing) if statement below creates the CTER patches for all residues
    if char in alphabit:
        op1.write('PRES cap_'+char+'3      -1.00 ! standard C-terminus\n')
        op1.write('GROUP                  ! use in generate statement\n')
        op1.write('ATOM C__'+charUC+' CC      0.34 !   OT2(-)\n')
        op1.write('ATOM OT1'+charUC+' OC     -0.67 !  /\n')
        op1.write('ATOM OT2'+charUC+' OC     -0.67 ! -C\n')
        op1.write('DELETE ATOM O__'+charUC+'       !  '+str('\\')+str('\\')+'\n')
        op1.write('BOND C__'+charUC+' OT2'+charUC+'         !   OT1\n')
        op1.write('DOUBLE C__'+charUC+' OT1'+charUC+'\n')
        op1.write('IMPR C__'+charUC+' CA_'+charUC+' OT2'+charUC+' OT1'+charUC+'\n')
        op1.write('ACCEPTOR OT1'+charUC+' C__'+charUC+'\n')
        op1.write('ACCEPTOR OT2'+charUC+' C__'+charUC+'\n')
        op1.write('IC N__'+charUC+' CA_'+charUC+' C__'+charUC+'  OT2'+charUC+'  0.0000  0.0000  180.0000  0.0000  0.0000\n')
        op1.write('IC OT2'+charUC+' CA_'+charUC+' *C__'+charUC+' OT1'+charUC+'  0.0000  0.0000  180.0000  0.0000  0.0000\n')
        op1.write('\n')

#for char in alphabit:
#    charUC = char.upper()
    ##The two if/else statements below create ACE/ACP patches for all residues
    if char == 'p':
        op1.write('PRES cap_p4       0.00 ! acetylated N-terminus for proline\n')
        op1.write('                       ! do NOT use to create dipeptide, see ACPD\n')
        op1.write('GROUP                  ! use in generate statement\n')
        op1.write('ATOM CAYP CT3    -0.27 !\n')
        op1.write('ATOM HY1P HA3     0.09 ! HY1 HY2 HY3\n')
        op1.write('ATOM HY2P HA3     0.09 !    \ | /\n')
        op1.write('ATOM HY3P HA3     0.09 !     CAY\n')
        op1.write('GROUP                  !      |\n')
        op1.write('ATOM CY_P C       0.51 !      CY=OY\n')
        op1.write('ATOM OY_P O      -0.51 !      |\n')
        op1.write('                       !\n')
        op1.write('BOND CY_P CAYP CY_P N__P CAYP HY1P CAYP HY2P CAYP HY3P\n')
        op1.write('DOUBLE OY_P CY_P\n')
        op1.write('IMPR CY_P CAYP N__P OY_P\n')
        op1.write('IMPR N__P CY_P CA_P CD_P\n')
        op1.write('CMAP CY_P N__P CA_P C__P N__P CA_P C__P +N\n')
        op1.write('ACCEPTOR OY_P CY_P\n')
        op1.write('IC CY_P N__P CA_P  C__P  0.0000  0.0000  -60.0000  0.0000  0.0000\n')
        op1.write('IC CY_P CA_P *N__P CD_P  0.0000  0.0000  180.0000  0.0000  0.0000\n')
        op1.write('IC CAYP CY_P N__P  CA_P  0.0000  0.0000  180.0000  0.0000  0.0000\n')
        op1.write('IC N__P CAYP *CY_P OY_P  0.0000  0.0000  180.0000  0.0000  0.0000\n')
        op1.write('IC OY_P CY_P CAYP  HY1P  0.0000  0.0000  180.0000  0.0000  0.0000\n')
        op1.write('IC OY_P CY_P CAYP  HY2P  0.0000  0.0000   60.0000  0.0000  0.0000\n')
        op1.write('IC OY_P CY_P CAYP  HY3P  0.0000  0.0000  -60.0000  0.0000  0.0000\n')
        op1.write('\n')
        
    else:
        op1.write('PRES cap_'+char+'4       0.00 ! acetylated N-terminus\n')
        op1.write('                       ! do NOT use to create dipeptides, see ACED\n')
        op1.write('GROUP                  ! use in generate statement\n')
        op1.write('ATOM CAY'+charUC+' CT3    -0.27 !\n')
        op1.write('ATOM HY1'+charUC+' HA3     0.09 ! HY1 HY2 HY3\n')
        op1.write('ATOM HY2'+charUC+' HA3     0.09 !    \\ | /\n')
        op1.write('ATOM HY3'+charUC+' HA3     0.09 !     CAY\n')
        op1.write('GROUP                  !      |\n')
        op1.write('ATOM CY_'+charUC+' C       0.51 !      CY=OY\n')
        op1.write('ATOM OY_'+charUC+' O      -0.51 !      |\n')
        op1.write('                       !\n')
        op1.write('BOND CY_'+charUC+' CAY'+charUC+' CY_'+charUC+' N__'+charUC+' CAY'+charUC+' HY1'+charUC+' CAY'+charUC+' HY2'+charUC+' CAY'+charUC+' HY3'+charUC+'\n')
        op1.write('DOUBLE OY_'+charUC+' CY_'+charUC+'\n')
        op1.write('IMPR CY_'+charUC+' CAY'+charUC+' N__'+charUC+' OY_'+charUC+'\n')
        op1.write('IMPR N__'+charUC+' CY_'+charUC+' CA_'+charUC+' HN_'+charUC+'\n')
        op1.write('CMAP CY_'+charUC+' N__'+charUC+' CA_'+charUC+' C__'+charUC+' N__'+charUC+' CA_'+charUC+' C__'+charUC+' +N\n')
        op1.write('ACCEPTOR OY_'+charUC+' CY_'+charUC+'\n')
        op1.write('IC CY_'+charUC+' N__'+charUC+' CA_'+charUC+'  C__'+charUC+'  0.0000  0.0000  -60.0000  0.0000  0.0000\n')
        op1.write('IC CY_'+charUC+' CA_'+charUC+' *N__'+charUC+' HN_'+charUC+'  0.0000  0.0000  180.0000  0.0000  0.0000\n')
        op1.write('IC CAY'+charUC+' CY_'+charUC+' N__'+charUC+'  CA_'+charUC+'  0.0000  0.0000  180.0000  0.0000  0.0000\n')
        op1.write('IC N__'+charUC+' CAY'+charUC+' *CY_'+charUC+' OY_'+charUC+'  0.0000  0.0000  180.0000  0.0000  0.0000\n')
        op1.write('IC OY_'+charUC+' CY_'+charUC+' CAY'+charUC+'  HY1'+charUC+'  0.0000  0.0000  180.0000  0.0000  0.0000\n')
        op1.write('IC OY_'+charUC+' CY_'+charUC+' CAY'+charUC+'  HY2'+charUC+'  0.0000  0.0000   60.0000  0.0000  0.0000\n')
        op1.write('IC OY_'+charUC+' CY_'+charUC+' CAY'+charUC+'  HY3'+charUC+'  0.0000  0.0000  -60.0000  0.0000  0.0000\n')
        op1.write('\n')

#for char in alphabit:
#    charUC = char.upper()
    ##The (uncessary but aestetichally pleasing) if statement below creates the CT3 patches for all residues
    if char in alphabit:
        op1.write('PRES cap_'+char+'5       0.00 ! N-Methylamide C-terminus\n')
        op1.write('GROUP                  ! use in generate statement\n')
        op1.write('ATOM C__'+charUC+' C       0.51 !\n')
        op1.write('ATOM O__'+charUC+' O      -0.51 !      |\n')
        op1.write('GROUP                  !      C=O\n')
        op1.write('ATOM NT_'+charUC+' NH1    -0.47 !      | \n')
        op1.write('ATOM HNT'+charUC+' H       0.31 !      NT-HNT\n')
        op1.write('ATOM CAT'+charUC+' CT3    -0.11 !      |\n')
        op1.write('ATOM HT1'+charUC+' HA3     0.09 ! HT1-CAT-HT3\n')
        op1.write('ATOM HT2'+charUC+' HA3     0.09 !      | \n')
        op1.write('ATOM HT3'+charUC+' HA3     0.09 !     HT2\n')
        op1.write('                       !\n')
        op1.write('BOND C__'+charUC+' NT_'+charUC+' NT_'+charUC+' HNT'+charUC+' NT_'+charUC+' CAT'+charUC+' CAT'+charUC+' HT1'+charUC+' CAT'+charUC+' HT2'+charUC+' CAT'+charUC+' HT3'+charUC+'\n')
        op1.write('IMPR NT_'+charUC+' C__'+charUC+' CAT'+charUC+' HNT'+charUC+' C__'+charUC+' CA_'+charUC+' NT_'+charUC+' O__'+charUC+'\n')
        op1.write('CMAP -C N__'+charUC+' CA_'+charUC+' C__'+charUC+' N__'+charUC+' CA_'+charUC+' C__'+charUC+' NT_'+charUC+'\n')
        op1.write('!CMAP CY  N  CA  C   N  CA  C  NT\n')
        op1.write('DONOR HNT'+charUC+' NT_'+charUC+'\n')
        op1.write('IC N__'+charUC+' CA_'+charUC+' C__'+charUC+'  NT_'+charUC+'  0.0000  0.0000  180.0000  0.0000  0.0000\n')
        op1.write('IC NT_'+charUC+' CA_'+charUC+' *C__'+charUC+' O__'+charUC+'  0.0000  0.0000  180.0000  0.0000  0.0000\n')
        op1.write('IC C__'+charUC+' CAT'+charUC+' *NT_'+charUC+' HNT'+charUC+'  0.0000  0.0000  180.0000  0.0000  0.0000\n')
        op1.write('IC CA_'+charUC+' C__'+charUC+' NT_'+charUC+'  CAT'+charUC+'  0.0000  0.0000  180.0000  0.0000  0.0000\n')
        op1.write('IC C__'+charUC+' NT_'+charUC+' CAT'+charUC+'  HT1'+charUC+'  0.0000  0.0000   60.0000  0.0000  0.0000\n')
        op1.write('IC C__'+charUC+' NT_'+charUC+' CAT'+charUC+'  HT2'+charUC+'  0.0000  0.0000  180.0000  0.0000  0.0000\n')
        op1.write('IC C__'+charUC+' NT_'+charUC+' CAT'+charUC+'  HT3'+charUC+'  0.0000  0.0000  -60.0000  0.0000  0.0000\n')
        op1.write('\n')

for char in alphabit:
    charUC = char.upper()
    op1.write('PRES l_'+char+'00         0.00\n')
    op1.write('IMPR N -C__'+charUC+' CA HN\n')
    op1.write('CMAP -C__'+charUC+' N CA C N CA C +N\n')
    op1.write('\n')

for char in alphabit:
    charUC = char.upper()
    op1.write('PRES l_'+char+'10         0.00\n')
    op1.write('IMPR N -C__'+charUC+' CA CD\n')
    op1.write('CMAP -C__'+charUC+' N CA C N CA C +N\n')
    op1.write('\n')

for char in alphabit:
    charUC = char.upper()
    op1.write('PRES l_'+char+'03         0.00\n')
    op1.write('IMPR N -C__'+charUC+' CA HN\n')
    op1.write('\n')

for char in alphabit:
    charUC = char.upper()
    op1.write('PRES l_'+char+'13         0.00\n')
    op1.write('IMPR N -C__'+charUC+' CA CD\n')
    op1.write('\n')

for char in alphabit:
    charUC = char.upper()
    op1.write('PRES l_'+char+'05         0.00\n')
    op1.write('IMPR N -C__'+charUC+' CA HN\n')
    op1.write('CMAP -C__'+charUC+' N CA C N CA C NT\n')
    op1.write('\n')

for char in alphabit:
    charUC = char.upper()
    op1.write('PRES l_'+char+'15         0.00\n')
    op1.write('IMPR N -C__'+charUC+' CA CD\n')
    op1.write('CMAP -C__'+charUC+' N CA C N CA C NT\n')
    op1.write('\n')

for char in alphabit:
    charUC = char.upper()
    op1.write('PRES l_00'+char+'         0.00\n')
    op1.write('IMPR C CA +N__'+charUC+' O\n')
    op1.write('CMAP -C N CA C N CA C +N__'+charUC+'\n')
    op1.write('\n')

for char in alphabit:
    charUC = char.upper()
    op1.write('PRES l_20'+char+'         0.00\n')
    op1.write('IMPR C CA +N__'+charUC+' O\n')
    op1.write('\n')

for char in alphabit:
    charUC = char.upper()
    op1.write('PRES l_40'+char+'         0.00\n')
    op1.write('IMPR C CA +N__'+charUC+' O\n')
    op1.write('CMAP CY N CA C N CA C +N__'+charUC+'\n')
    op1.write('\n')

for char1 in alphabit:
    char1UC = char1.upper()
    for char2 in alphabit:
        char2UC = char2.upper()
        op1.write('PRES l_'+char1+char2+'0         0.00\n')
        op1.write('BOND -C__'+char1UC+' N__'+char2UC+'\n')
        if char2UC == 'P':
            op1.write('IMPR N__'+char2UC+' -C__'+char1UC+' CA_'+char2UC+' CD_'+char2UC+'\n')
        else:
            op1.write('IMPR N__'+char2UC+' -C__'+char1UC+' CA_'+char2UC+' HN_'+char2UC+'\n')
        op1.write('CMAP -C__'+char1UC+' N__'+char2UC+' CA_'+char2UC+' C__'+char2UC+' N__'+char2UC+' CA_'+char2UC+' C__'+char2UC+' +N\n')
        op1.write('\n')

for char1 in alphabit:
    char1UC = char1.upper()
    for char2 in alphabit:
        char2UC = char2.upper()
        op1.write('PRES l_'+char1+char2+'3         0.00\n')
        op1.write('BOND -C__'+char1UC+' N__'+char2UC+'\n')
        if char2UC == 'P':
            op1.write('IMPR N__'+char2UC+' -C__'+char1UC+' CA_'+char2UC+' CD_'+char2UC+'\n')
        else:
            op1.write('IMPR N__'+char2UC+' -C__'+char1UC+' CA_'+char2UC+' HN_'+char2UC+'\n')
        op1.write('\n')

for char1 in alphabit:
    char1UC = char1.upper()
    for char2 in alphabit: 
        char2UC = char2.upper()
        op1.write('PRES l_'+char1+char2+'5         0.00\n')
        op1.write('BOND -C__'+char1UC+' N__'+char2UC+'\n')
        if char2UC == 'P':
            op1.write('IMPR N__'+char2UC+' -C__'+char1UC+' CA_'+char2UC+' CD_'+char2UC+'\n')
        else:
            op1.write('IMPR N__'+char2UC+' -C__'+char1UC+' CA_'+char2UC+' HN_'+char2UC+'\n')
        op1.write('CMAP -C__'+char1UC+' N__'+char2UC+' CA_'+char2UC+' C__'+char2UC+' N__'+char2UC+' CA_'+char2UC+' C__'+char2UC+' NT_'+char2UC+'\n')
        op1.write('\n')

for char1 in alphabit:
    char1UC = char1.upper()
    for char2 in alphabit:
        char2UC = char2.upper()
        op1.write('PRES l_'+char1+'0'+char2+'         0.00\n')
        op1.write('CMAP -C__'+char1UC+' N CA C N CA C +N__'+char2UC+'\n')
        op1.write('\n')

for char1 in alphabit:
    char1UC = char1.upper()
    for char2 in alphabit:
        char2UC = char2.upper()
        op1.write('PRES l_0'+char1+char2+'         0.00\n')
        op1.write('IMPR C__'+char1UC+' CA_'+char1UC+' +N__'+char2UC+' O__'+char1UC+'\n')
        op1.write('CMAP -C N__'+char1UC+' CA_'+char1UC+' C__'+char1UC+' N__'+char1UC+' CA_'+char1UC+' C__'+char1UC+' +N__'+char2UC+'\n')
        op1.write('\n')

for char1 in alphabit:
    char1UC = char1.upper()
    for char2 in alphabit:
        char2UC = char2.upper()
        op1.write('PRES l_2'+char1+char2+'         0.00\n')
        op1.write('IMPR C__'+char1UC+' CA_'+char1UC+' +N__'+char2UC+' O__'+char1UC+'\n')
        op1.write('\n')

for char1 in alphabit: 
    char1UC = char1.upper()
    for char2 in alphabit:
        char2UC = char2.upper()
        op1.write('PRES l_4'+char1+char2+'         0.00\n')
        op1.write('IMPR C__'+char1UC+' CA_'+char1UC+' +N__'+char2UC+' O__'+char1UC+'\n')
        op1.write('CMAP CY_'+char1UC+' N__'+char1UC+' CA_'+char1UC+' C__'+char1UC+' N__'+char1UC+' CA_'+char1UC+' C__'+char1UC+' +N__'+char2UC+'\n')
        op1.write('\n')

for char1 in alphabit:
    char1UC = char1.upper()
    for char2 in alphabit:
        char2UC = char2.upper()
        for char3 in alphabit:
            char3UC = char3.upper()
            op1.write('PRES l_'+char1+char2+char3+'         0.00\n')
            op1.write('CMAP -C__'+char1UC+' N__'+char2UC+' CA_'+char2UC+' C__'+char2UC+' N__'+char2UC+' CA_'+char2UC+' C__'+char2UC+' +N__'+char3UC+'\n')
            op1.write('\n')

op1.write('end\n')
op1.write('return\n')
op1.close()
