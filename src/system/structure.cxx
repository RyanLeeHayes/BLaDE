#include <string.h>

#include "system/system.h"
#include "system/structure.h"
#include "main/defines.h"
#include "io/io.h"

void parse_structure(char *line,System *system)
{
  char token[MAXLENGTHSTRING];
  FILE *fp;

  // All routines except reset need a strcture, so just initialize it and save time.
  if (system->structure==NULL) {
    system->structure=new Structure();
  }

  io_nexta(line,token);
  if (strcmp(token,"reset")==0) {
    if (system->structure) {
      delete(system->structure);
      system->structure=NULL;
    }
  } else if (strcmp(token,"file")==0) {
    io_nexta(line,token);
    if (strcmp(token,"psf")==0) {
      io_nexta(line,token);
      fp=fpopen(token,"r");
      system->structure->add_structure_psf_file(fp);
      fclose(fp);
    } else {
      fatal(__FILE__,__LINE__,"Unsupported structure file format: %s\n",token);
    }
  } else if (strcmp(token,"setup")==0) {
    system->structure->setup(system);
  } else if (strcmp(token,"print")==0) {
    system->structure->dump();
  } else {
    fatal(__FILE__,__LINE__,"Unrecognized structure token: %s\n",token);
  }
}

void Structure::add_structure_psf_file(FILE *fp)
{
  char line[MAXLENGTHSTRING];
  char token[MAXLENGTHSTRING];
  int i,j,k;

  // "read" header"
  fgets(line, MAXLENGTHSTRING, fp);
  io_nexta(line,token);
  if (strcmp(token,"PSF")!=0) {
    fatal(__FILE__,__LINE__,"First line of PSF must start with PSF\n");
  }

  // "Read" title
  fgets(line, MAXLENGTHSTRING, fp);
  j=io_nexti(line,fp,"psf number of title lines");
  for (i=0; i<j; i++) {
    fgets(line, MAXLENGTHSTRING, fp);
  }

  // Read atoms
  fgets(line, MAXLENGTHSTRING, fp);
  atomCount=io_nexti(line,fp,"psf number of atoms");
  atomList.clear();
  atomList.reserve(atomCount);
  for (i=0; i<atomCount; i++) {
    fgets(line, MAXLENGTHSTRING, fp);
    struct AtomStructure at;
    at.atomIdx=io_nexti(line)-1;
    if (at.atomIdx!=i) {
      fatal(__FILE__,__LINE__,"Found atom %d when atom %d expected\n",at.atomIdx,i);
    }
    at.segName=io_nexts(line);
    at.resIdx=io_nexti(line);
    at.resName=io_nexts(line);
    at.atomName=io_nexts(line);
    at.atomTypeName=io_nexts(line);
    at.charge=io_nextf(line);
    at.mass=io_nextf(line);
    atomList.emplace_back(at);
  }

  // Read bonds
  fgets(line, MAXLENGTHSTRING, fp);
  bondCount=io_nexti(line,fp,"psf number of bonds");
  bondList.clear();
  bondList.reserve(bondCount);
  fgets(line, MAXLENGTHSTRING, fp);
  for (i=0; i<bondCount; i++) {
    struct Int2 bond;
    for (j=0; j<2; j++) {
      k=io_nexti(line,fp,"psf bond atom")-1;
      if (k>=atomCount || k<0) {
        fatal(__FILE__,__LINE__,"Atom %d in bond %d is out of range\n",k,i);
      }
      bond.i[j]=k;
    }
    bondList.emplace_back(bond);
  }
  
  // Read angles
  fgets(line, MAXLENGTHSTRING, fp);
  angleCount=io_nexti(line,fp,"psf number of angles");
  angleList.clear();
  angleList.reserve(angleCount);
  fgets(line, MAXLENGTHSTRING, fp);
  for (i=0; i<angleCount; i++) {
    struct Int3 angle;
    for (j=0; j<3; j++) {
      k=io_nexti(line,fp,"psf angle atom")-1;
      if (k>=atomCount || k<0) {
        fatal(__FILE__,__LINE__,"Atom %d in angle %d is out of range\n",k,i);
      }
      angle.i[j]=k;
    }
    angleList.emplace_back(angle);
  }
  
  // Read dihes
  fgets(line, MAXLENGTHSTRING, fp);
  diheCount=io_nexti(line,fp,"psf number of dihedrals");
  diheList.clear();
  diheList.reserve(diheCount);
  fgets(line, MAXLENGTHSTRING, fp);
  for (i=0; i<diheCount; i++) {
    struct Int4 dihe;
    for (j=0; j<4; j++) {
      k=io_nexti(line,fp,"psf dihedral atom")-1;
      if (k>=atomCount || k<0) {
        fatal(__FILE__,__LINE__,"Atom %d in dihedral %d is out of range\n",k,i);
      }
      dihe.i[j]=k;
    }
    diheList.emplace_back(dihe);
  }
  
  // Read imprs
  fgets(line, MAXLENGTHSTRING, fp);
  imprCount=io_nexti(line,fp,"psf number of impropers");
  imprList.clear();
  imprList.reserve(imprCount);
  fgets(line, MAXLENGTHSTRING, fp);
  for (i=0; i<imprCount; i++) {
    struct Int4 impr;
    for (j=0; j<4; j++) {
      k=io_nexti(line,fp,"psf improper dih atom")-1;
      if (k>=atomCount || k<0) {
        fatal(__FILE__,__LINE__,"Atom %d in improper %d is out of range\n",k,i);
      }
      impr.i[j]=k;
    }
    imprList.emplace_back(impr);
  }
  
  // Ignore donors
  fgets(line, MAXLENGTHSTRING, fp);
  j=io_nexti(line,fp,"psf number of donors");
  for (i=0; i<2*j; i++) {
    io_nexti(line,fp,"psf donor atom");
  }
  
  // Ignore acceptors
  fgets(line, MAXLENGTHSTRING, fp);
  j=io_nexti(line,fp,"psf number of acceptors");
  for (i=0; i<2*j; i++) {
    io_nexti(line,fp,"psf acceptor atom");
  }
  
  // Not even sure what this section is...
  fgets(line, MAXLENGTHSTRING, fp);
  j=io_nexti(line,fp,"psf nnb???");
  for (i=0; i<atomCount; i++) {
    io_nexti(line,fp,"psf nnb???");
  }

  // Or this one...
  fgets(line, MAXLENGTHSTRING, fp);
  j=io_nexti(line,fp,"psf ngrp???");
  for (i=0; i<3*j; i++) {
    io_nexti(line,fp,"psf ngrp???");
  }

  // OR this one...
  fgets(line, MAXLENGTHSTRING, fp);
  j=io_nexti(line,fp,"psf molnt???");
  for (i=0; i<atomCount; i++) {
    io_nexti(line,fp,"psf molnt???");
  }

  // ignore lone pairs
  fgets(line, MAXLENGTHSTRING, fp);
  i=io_nexti(line,fp,"psf lone pairs");
  j=io_nexti(line,fp,"psf lone pair hydrogrens");
  if (i!=0 || j!=0) {
    fatal(__FILE__,__LINE__,"Program is not set up to treat lone pairs. Found NUMLP=%d NUMLPH=%d in psf\n",i,j);
  }
  
  // Read cmaps
  fgets(line, MAXLENGTHSTRING, fp);
  cmapCount=io_nexti(line,fp,"psf number of cmaps");
  cmapList.clear();
  cmapList.reserve(cmapCount);
  fgets(line, MAXLENGTHSTRING, fp);
  for (i=0; i<cmapCount; i++) {
    struct Int8 cmap;
    for (j=0; j<8; j++) {
      k=io_nexti(line,fp,"psf cmap atom")-1;
      if (k>=atomCount || k<0) {
        fatal(__FILE__,__LINE__,"Atom %d in cmap %d is out of range\n",k,i);
      }
      cmap.i[j]=k;
    }
    cmapList.emplace_back(cmap);
  }
}

void Structure::setup(System *system)
{
  int i,j;
  Parameters *param=system->parameters;

  atomCount=atomList.size();
  bondCount=bondList.size();
  angleCount=angleList.size();
  diheCount=diheList.size();
  imprCount=imprList.size();
  cmapCount=cmapList.size();

  if (atomTypeIdx) free(atomTypeIdx);
  atomTypeIdx=(int*)calloc(atomCount,sizeof(int));
  if (charge) free(charge);
  charge=(real*)calloc(atomCount,sizeof(real));
  if (mass) free(mass);
  mass=(real*)calloc(atomCount,sizeof(real));

  bonds=(struct BondStructure*)calloc(bondCount,sizeof(struct BondStructure));
  angles=(struct AngleStructure*)calloc(angleCount,sizeof(struct AngleStructure));
  dihes=(struct DiheStructure*)calloc(diheCount,sizeof(struct DiheStructure));
  imprs=(struct ImprStructure*)calloc(imprCount,sizeof(struct ImprStructure));
  cmaps=(struct CmapStructure*)calloc(cmapCount,sizeof(struct CmapStructure));

  for (i=0; i<atomCount; i++) {
    struct AtomStructure at=atomList[i];
    param->require_type_name(at.atomTypeName,"searching for atom types in structure setup");
    atomTypeIdx[i]=param->atomTypeMap[at.atomTypeName];
    charge[i]=at.charge;
    mass[i]=param->atomMass[at.atomTypeName];
  }

  for (i=0; i<bondCount; i++) {
    TypeName2 type;
    struct BondStructure bond;
    struct BondParameter bp;
    for (j=0; j<2; j++) {
      bond.idx[j]=bondList[i].i[j];
      type.t[j]=atomList[bond.idx[j]].atomTypeName;
    }
    if (param->bondParameter.count(type)==0) {
      fatal(__FILE__,__LINE__,"No bond parameter for %6d(%6s) %6d(%6s)\n",bond.idx[0],type.t[0].c_str(),bond.idx[1],type.t[1].c_str());
    }
    bp=param->bondParameter[type];
    bond.kb=bp.kb;
    bond.b0=bp.b0;
    bonds[i]=bond;
  }
    
  for (i=0; i<angleCount; i++) {
    TypeName3 type;
    struct AngleStructure angle;
    struct AngleParameter ap;
    for (j=0; j<3; j++) {
      angle.idx[j]=angleList[i].i[j];
      type.t[j]=atomList[angle.idx[j]].atomTypeName;
    }
    if (param->angleParameter.count(type)==0) {
      fatal(__FILE__,__LINE__,"No angle parameter for %6d(%6s) %6d(%6s) %6d(%6s)\n",angle.idx[0],type.t[0].c_str(),angle.idx[1],type.t[1].c_str(),angle.idx[2],type.t[2].c_str());
    }
    ap=param->angleParameter[type];
    angle.kangle=ap.kangle;
    angle.angle0=ap.angle0;
#warning "Urey Bradley terms are not separated out, may be inefficient.
    angle.kureyb=ap.kureyb;
    angle.ureyb0=ap.ureyb0;
    angles[i]=angle;
  }
    
  for (i=0; i<diheCount; i++) {
    TypeName4 type,typx;
    struct DiheStructure dihe;
    std::vector<struct DiheParameter> dp;
    for (j=0; j<4; j++) {
      dihe.idx[j]=diheList[i].i[j];
      type.t[j]=atomList[dihe.idx[j]].atomTypeName;
    }
    if (param->diheParameter.count(type)==1) {
      dp=param->diheParameter[type];
    } else {
      typx=type;
      typx.t[0]="X";
      typx.t[3]="X";
      if (param->diheParameter.count(typx)==1) {
        dp=param->diheParameter[typx];
      } else {
        fatal(__FILE__,__LINE__,"No dihe parameter for %6d(%6s) %6d(%6s) %6d(%6s) %6d(%6s)\n",dihe.idx[0],type.t[0].c_str(),dihe.idx[1],type.t[1].c_str(),dihe.idx[2],type.t[2].c_str(),dihe.idx[3],type.t[3].c_str());
      }
    }
#warning "Serious error: ignoring higher order dihedral terms for now"
    dihe.kdih=dp[0].kdih;
    dihe.ndih=dp[0].ndih;
    dihe.dih0=dp[0].dih0;
    dihes[i]=dihe;
  }

  for (i=0; i<imprCount; i++) {
    TypeName4 type,typx;
    struct ImprStructure impr;
    struct ImprParameter ip;
    for (j=0; j<4; j++) {
      impr.idx[j]=imprList[i].i[j];
      type.t[j]=atomList[impr.idx[j]].atomTypeName;
    }
    if (param->imprParameter.count(type)==1) { // 1st ABCD
      ip=param->imprParameter[type];
    } else {
      typx=type;
      typx.t[1]="X";
      typx.t[2]="X";
      if (param->imprParameter.count(typx)==1) { // 2nd AXXD
        ip=param->imprParameter[typx];
      } else {
        typx=type;
        typx.t[0]="X";
        if (param->imprParameter.count(typx)==1) { // 3rd XBCD
          ip=param->imprParameter[typx];
        } else {
          typx.t[3]="X";
          if (param->imprParameter.count(typx)==1) { // 4th XBCX
            ip=param->imprParameter[typx];
          } else {
            typx=type;
            typx.t[0]="X";
            typx.t[1]="X";
            if (param->imprParameter.count(typx)==1) { // 5th AXXD
              ip=param->imprParameter[typx];
            } else {
              fatal(__FILE__,__LINE__,"No impr parameter for %6d(%6s) %6d(%6s) %6d(%6s) %6d(%6s)\n",impr.idx[0],type.t[0].c_str(),impr.idx[1],type.t[1].c_str(),impr.idx[2],type.t[2].c_str(),impr.idx[3],type.t[3].c_str());
            }
          }
        }
      }
    }
    impr.kimp=ip.kimp;
    impr.imp0=ip.imp0;
    imprs[i]=impr;
  }

#warning "Missing CMAP terms"
#warning "Missing nonbonded terms"

  fprintf(stdout,"Implement nonbonded structures too\n");
}

void Structure::dump()
{
  fprintf(stdout,"%s:%d IMPLEMENT Structure::dump function.\n",__FILE__,__LINE__);
}
