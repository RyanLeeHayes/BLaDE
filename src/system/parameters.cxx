#include <stdio.h>
#include <string.h>
#include <math.h>

#include <string>

#include "main/defines.h"
#include "system/system.h"
#include "system/parameters.h"
#include "io/io.h"

// NAMD reference for CHARMM file format:
// https://www.ks.uiuc.edu/Training/Tutorials/namd/namd-tutorial-unix-html/node25.html
// Consult when functional form is not listed in parameter file

void parse_parameters(char *line,System *system)
{
  char token[MAXLENGTHSTRING];
  FILE *fp;

  io_nexta(line,token);
  if (strcmp(token,"reset")==0) {
    if (system->parameters) {
      delete(system->parameters);
      system->parameters=NULL;
    }
  } else if (strcmp(token,"file")==0) {
    if (system->parameters==NULL) {
      system->parameters=new Parameters();
    }
    io_nexta(line,token);
    fprintf(stdout,"DEBUG: fnm: %s\n",token);
    fp=fpopen(token,"r");
    system->parameters->add_parameter_file(fp);
  } else if (strcmp(token,"print")==0) {
    system->parameters->dump();
  } else {
    fatal(__FILE__,__LINE__,"Unrecognized token: %s\n",token); // FIXIT add token name
  }
}

void Parameters::add_parameter_file(FILE *fp)
{
  char line[MAXLENGTHSTRING];
  char token[MAXLENGTHSTRING];

  while (fgets(line, MAXLENGTHSTRING, fp) != NULL) {
    if (line[0]=='*') {
      fprintf(stdout,"TITLE> %s",line);
      continue;
    }
    io_nexta(line,token);
    if (strcmp(token,"")==0) {
      ;
    } else if (strncmp(token,"ATOMS",4)==0) {
      add_parameter_atoms(fp);
    } else if (strncmp(token,"BONDS",4)==0) {
      add_parameter_bonds(fp);
    } else if (strncmp(token,"ANGLES",4)==0) {
      add_parameter_angles(fp);
    } else if (strncmp(token,"DIHEDRALS",4)==0) {
      add_parameter_dihs(fp);
    } else if (strncmp(token,"IMPROPERS",4)==0) {
      add_parameter_imps(fp);
    } else if (strncmp(token,"CMAP",4)==0) {
      add_parameter_cmaps(fp);
    } else if (strncmp(token,"NONBONDED",4)==0) {
      add_parameter_nonbondeds(line,fp);
    } else if (strncmp(token,"NBFIX",4)==0) {
      add_parameter_nbfixs(fp);
    } else if (strcmp(token,"END")==0) {
      return;
    } else {
      fprintf(stdout,"Unparsed line> %s %s",token,line);
    }
  }
}

void Parameters::add_parameter_atoms(FILE *fp)
{
  fpos_t fp_pos;
  char line[MAXLENGTHSTRING];
  char token[MAXLENGTHSTRING];
  int i;
  std::string name;
  double m;

  fgetpos(fp,&fp_pos);
  while (fgets(line, MAXLENGTHSTRING, fp) != NULL) {
    io_nexta(line,token);
    if (strcmp(token,"")==0) {
      ;
    } else if (strcmp(token,"MASS")==0) {
      i=io_nexti(line); // Ignore index
      name=io_nexts(line);
      m=io_nextf(line);
      // parameters->atomType.insert(name,parameters->atomTypeCount);
      atomTypeMap[name]=atomTypeCount;
      atomType.emplace_back(name);
      atomMass.emplace_back(m);
      atomTypeCount++;
    } else {
      // This is part of next section. Back up and return.
      fsetpos(fp,&fp_pos);
      return;
    }

    fgetpos(fp,&fp_pos);
  }
}

void Parameters::add_parameter_bonds(FILE *fp)
{
  fpos_t fp_pos;
  char line[MAXLENGTHSTRING];
  std::string iname;
  struct BondParameter bp;

  fgetpos(fp,&fp_pos);
  while (fgets(line, MAXLENGTHSTRING, fp) != NULL) {
    iname=io_nexts(line);
    if (strcmp(iname.c_str(),"")==0) {
      ;
    } else if (atomTypeMap.count(iname)==1) {
      bp.i=atomTypeMap[iname];
#warning "No sanity check"
      bp.j=atomTypeMap[io_nexts(line)];
      bp.kb=(2.0*KCAL_MOL/(ANGSTROM*ANGSTROM))*io_nextf(line);
      bp.b0=ANGSTROM*io_nextf(line);
      bondParameter.emplace_back(bp);
      bondParameterCount++;
    } else {
      // This is part of next section. Back up and return.
      fsetpos(fp,&fp_pos);
      return;
    }

    fgetpos(fp,&fp_pos);
  }
}

void Parameters::add_parameter_angles(FILE *fp)
{
  fpos_t fp_pos;
  char line[MAXLENGTHSTRING];
  std::string iname;
  struct AngleParameter ap;

  fgetpos(fp,&fp_pos);
  while (fgets(line, MAXLENGTHSTRING, fp) != NULL) {
    iname=io_nexts(line);
    if (strcmp(iname.c_str(),"")==0) {
      ;
    } else if (atomTypeMap.count(iname)==1) {
      ap.i=atomTypeMap[iname];
#warning "No sanity check"
      ap.j=atomTypeMap[io_nexts(line)];
      ap.k=atomTypeMap[io_nexts(line)];
      ap.kangle=(2.0*KCAL_MOL)*io_nextf(line);
      ap.angle0=DEGREES*io_nextf(line);
      ap.kureyb=(2.0*KCAL_MOL/(ANGSTROM*ANGSTROM))*io_nextf(line,0);
      ap.ureyb0=ANGSTROM*io_nextf(line,0);
      angleParameter.emplace_back(ap);
      angleParameterCount++;
    } else {
      // This is part of next section. Back up and return.
      fsetpos(fp,&fp_pos);
      return;
    }

    fgetpos(fp,&fp_pos);
  }
}

// V(dihedral) = Kdih(1 + cos(ndih(chi) - dih0))
void Parameters::add_parameter_dihs(FILE *fp)
{
  fpos_t fp_pos;
  char line[MAXLENGTHSTRING];
  std::string iname;
  struct DihParameter dp;

  fgetpos(fp,&fp_pos);
  while (fgets(line, MAXLENGTHSTRING, fp) != NULL) {
    iname=io_nexts(line);
    if (strcmp(iname.c_str(),"")==0) {
      ;
    } else if (atomTypeMap.count(iname)==1) {
      dp.i=atomTypeMap[iname];
#warning "No sanity check"
      dp.j=atomTypeMap[io_nexts(line)];
      dp.k=atomTypeMap[io_nexts(line)];
      dp.l=atomTypeMap[io_nexts(line)];
      dp.kdih=KCAL_MOL*io_nextf(line);
      dp.ndih=io_nexti(line);
      dp.dih0=DEGREES*io_nextf(line);
      dihParameter.emplace_back(dp);
      dihParameterCount++;
    } else {
      // This is part of next section. Back up and return.
      fsetpos(fp,&fp_pos);
      return;
    }

    fgetpos(fp,&fp_pos);
  }
}

void Parameters::add_parameter_imps(FILE *fp)
{
  fpos_t fp_pos;
  char line[MAXLENGTHSTRING];
  std::string iname;
  struct ImpParameter ip;

  fgetpos(fp,&fp_pos);
  while (fgets(line, MAXLENGTHSTRING, fp) != NULL) {
    iname=io_nexts(line);
    if (strcmp(iname.c_str(),"")==0) {
      ;
    } else if (atomTypeMap.count(iname)==1) {
      ip.i=atomTypeMap[iname];
#warning "No sanity check"
      ip.j=atomTypeMap[io_nexts(line)];
      ip.k=atomTypeMap[io_nexts(line)];
      ip.l=atomTypeMap[io_nexts(line)];
      ip.kimp=(2.0*KCAL_MOL)*io_nextf(line);
      io_nexts(line);
      ip.imp0=DEGREES*io_nextf(line);
      impParameter.emplace_back(ip);
      impParameterCount++;
    } else {
      // This is part of next section. Back up and return.
      fsetpos(fp,&fp_pos);
      return;
    }

    fgetpos(fp,&fp_pos);
  }
}

void Parameters::add_parameter_cmaps(FILE *fp)
{
  fpos_t fp_pos;
  char line[MAXLENGTHSTRING];
  std::string iname;
  struct CmapParameter cp;
  int i,j;
  double k;
  char *escape;

  fgetpos(fp,&fp_pos);
  while (fgets(line, MAXLENGTHSTRING, fp) != NULL) {
    iname=io_nexts(line);
    if (strcmp(iname.c_str(),"")==0) {
      ;
    } else if (atomTypeMap.count(iname)==1) {
      cp.i1=atomTypeMap[iname];
#warning "No sanity check"
      cp.j1=atomTypeMap[io_nexts(line)];
      cp.k1=atomTypeMap[io_nexts(line)];
      cp.l1=atomTypeMap[io_nexts(line)];
      cp.i2=atomTypeMap[io_nexts(line)];
      cp.j2=atomTypeMap[io_nexts(line)];
      cp.k2=atomTypeMap[io_nexts(line)];
      cp.l2=atomTypeMap[io_nexts(line)];
#warning "Assume cmap size of 24"
      cp.ngrid=io_nexti(line);
      if (cp.ngrid!=24) {
        fatal(__FILE__,__LINE__,"Error: lazy coding, programmer assumed CMAP size would be 24 when it was %d\n",cp.ngrid);
      }

      for (i=0; i<24; i++) {
        for (j=0; j<24; j++) {
          for (k=-INFINITY; !((k=io_nextf(line,k))>-INFINITY); escape=fgets(line, MAXLENGTHSTRING, fp)) {
            // fprintf(stdout,"DEBUG i=%d j=%d k=%g line=%s\n",i,j,k,line);
            if (escape==NULL) {
              fatal(__FILE__,__LINE__,"CMAP error: could not read 24x24 matrix, stuck on element [%d][%d] at EOF\n",i,j);
            }
          }
          // fprintf(stdout,"DEBUG i=%d j=%d k=%g line=%s\n",i,j,k,line);
          cp.kcmap[i][j]=KCAL_MOL*k;
        }
      }
      cmapParameter.emplace_back(cp);
      cmapParameterCount++;
    } else {
      // This is part of next section. Back up and return.
      fsetpos(fp,&fp_pos);
      return;
    }

    fgetpos(fp,&fp_pos);
  }
}

/*
!
!V(Lennard-Jones) = Eps,i,j[(Rmin,i,j/ri,j)**12 - 2(Rmin,i,j/ri,j)**6]
!
!epsilon: kcal/mole, Eps,i,j = sqrt(eps,i * eps,j)
!Rmin/2: A, Rmin,i,j = Rmin/2,i + Rmin/2,j
!
!atom  ignored    epsilon      Rmin/2   ignored   eps,1-4       Rmin/2,1-4
!
*/
void Parameters::add_parameter_nonbondeds(char *line,FILE *fp)
{
  fpos_t fp_pos;
  // char line[MAXLENGTHSTRING];
  std::string iname;
  struct NonbondedParameter np;

  fgetpos(fp,&fp_pos);
  while (fgets(line, MAXLENGTHSTRING, fp) != NULL) {
    iname=io_nexts(line);
    if (strcmp(iname.c_str(),"")==0) {
      ;
    } else if (atomTypeMap.count(iname)==1) {
      np.i=atomTypeMap[iname];
      io_nextf(line);
      np.eps=io_nextf(line);
      np.sig=io_nextf(line);
      io_nextf(line,0);
      np.eps14=io_nextf(line,np.eps);
      np.sig14=io_nextf(line,np.sig);
      np.eps*=KCAL_MOL;
      np.sig*=ANGSTROM;
      np.eps14*=KCAL_MOL;
      np.sig14*=ANGSTROM;
      nonbondedParameter.emplace_back(np);
      nonbondedParameterCount++;
// Other acceptable tokens
#warning "Currently ignores nonbonded default parameters"
//      cutnb  14.0 ctofnb 12.0 ctonnb 10.0 eps 1.0 e14fac 1.0 wmin 1.5 
    } else if (strcmp(iname.c_str(),"cutnb")==0 ||
               strcmp(iname.c_str(),"ctofnb")==0 ||
               strcmp(iname.c_str(),"ctonnb")==0 ||
               strcmp(iname.c_str(),"eps")==0 ||
               strcmp(iname.c_str(),"e14fac")==0 ||
               strcmp(iname.c_str(),"wmin")==0) {
      ;
    } else {
      // This is part of next section. Back up and return.
      fsetpos(fp,&fp_pos);
      return;
    }

    fgetpos(fp,&fp_pos);
  }
}

void Parameters::add_parameter_nbfixs(FILE *fp)
{
  fpos_t fp_pos;
  char line[MAXLENGTHSTRING];
  std::string iname;
  struct NbfixParameter np;

  fgetpos(fp,&fp_pos);
  while (fgets(line, MAXLENGTHSTRING, fp) != NULL) {
    iname=io_nexts(line);
    if (strcmp(iname.c_str(),"")==0) {
      ;
    } else if (atomTypeMap.count(iname)==1) {
      np.i=atomTypeMap[iname];
      np.j=atomTypeMap[io_nexts(line)];
      np.eps=io_nextf(line);
      np.sig=io_nextf(line);
      np.eps14=io_nextf(line,np.eps);
      np.sig14=io_nextf(line,np.sig);
      np.eps*=KCAL_MOL;
      np.sig*=ANGSTROM;
      np.eps14*=KCAL_MOL;
      np.sig14*=ANGSTROM;
      nbfixParameter.emplace_back(np);
      nbfixParameterCount++;
    } else {
      // This is part of next section. Back up and return.
#warning "Unrecognized atom types cause function to return, rather than causing errors or being saved."
      fsetpos(fp,&fp_pos);
      return;
    }

    fgetpos(fp,&fp_pos);
  }
}


void Parameters::dump()
{
  int i;
  std::map<std::string,int>::iterator ii;
  std::string key;
  char tag[]="PRINT PARAMETERS>";

  fprintf(stdout,"%s atomTypeCount=%d\n",tag,atomTypeCount);
  fprintf(stdout,"%s\n",tag);

  for (ii=atomTypeMap.begin(); ii!=atomTypeMap.end(); ii++) {
    fprintf(stdout,"%s atomTypeMap[%5s]=%5d\n",tag,ii->first.c_str(),ii->second);
  }
  fprintf(stdout,"%s\n",tag);

  for (i=0; i<atomTypeCount; i++) {
    fprintf(stdout,"%s atomType[%5d]=%5s\n",tag,i,atomType[i].c_str());
  }
  fprintf(stdout,"%s\n",tag);

  for (i=0; i<atomTypeCount; i++) {
    fprintf(stdout,"%s atomMass[%5d]=%g\n",tag,i,atomMass[i]);
  }
  fprintf(stdout,"%s\n",tag);

  fprintf(stdout,"%s bondParameterCount=%d\n",tag,bondParameterCount);
  fprintf(stdout,"%s\n",tag);

  for (i=0; i<bondParameterCount; i++) {
    fprintf(stdout,"%s bondParameter[%5d]={i=%5d j=%5d kb=%g b0=%g}\n",tag,i,bondParameter[i].i,bondParameter[i].j,bondParameter[i].kb,bondParameter[i].b0);
  }
  fprintf(stdout,"%s\n",tag);

  fprintf(stdout,"%s angleParameterCount=%d\n",tag,angleParameterCount);
  fprintf(stdout,"%s\n",tag);

  for (i=0; i<angleParameterCount; i++) {
    struct AngleParameter ap=angleParameter[i];
    fprintf(stdout,"%s angleParameter[%5d]={i=%5d j=%5d k=%5d kangle=%g angle0=%g kureyb=%g ureyb0=%g}\n",tag,i,ap.i,ap.j,ap.k,ap.kangle,ap.angle0,ap.kureyb,ap.ureyb0);
  }
  fprintf(stdout,"%s\n",tag);

  fprintf(stdout,"%s dihParameterCount=%d\n",tag,dihParameterCount);
  fprintf(stdout,"%s\n",tag);

  for (i=0; i<dihParameterCount; i++) {
    struct DihParameter dp=dihParameter[i];
    fprintf(stdout,"%s dihParameter[%5d]={i=%5d j=%5d k=%5d l=%5d kdih=%g ndih=%d dih0=%g}\n",tag,i,dp.i,dp.j,dp.k,dp.l,dp.kdih,dp.ndih,dp.dih0);
  }
  fprintf(stdout,"%s\n",tag);

  fprintf(stdout,"%s impParameterCount=%d\n",tag,impParameterCount);
  fprintf(stdout,"%s\n",tag);

  for (i=0; i<impParameterCount; i++) {
    struct ImpParameter ip=impParameter[i];
    fprintf(stdout,"%s impParameter[%5d]={i=%5d j=%5d k=%5d l=%5d kimp=%g imp0=%g}\n",tag,i,ip.i,ip.j,ip.k,ip.l,ip.kimp,ip.imp0);
  }
  fprintf(stdout,"%s\n",tag);

  fprintf(stdout,"%s cmapParameterCount=%d\n",tag,cmapParameterCount);
  fprintf(stdout,"%s\n",tag);

  fprintf(stdout,"%s cmapParameter not printed\n",tag);
  fprintf(stdout,"%s nonbondedParameter not printed\n",tag);
  fprintf(stdout,"%s nbfixParameter not printed\n",tag);

  class twostrings {
    public:
    std::string s1,s2;

    bool operator==(const twostrings& a)
    {
      return (a.s1==s1 && a.s2==s2);
    }
  };

  struct twoints {
    int i1,i2;
  };

  twostrings s1,s2,s3;
  s1.s1="OT";
  s1.s2="HT";
  s2.s1="OT";
  s2.s2="HT";
  s3.s1="HT";
  s3.s2="HT";
  if (s1==s2) {
    fprintf(stdout,"s1==s2\n");
  }
  if (s1==s3) {
    fprintf(stdout,"s1==s3\n");
  } else {
    fprintf(stdout,"s1!=s3\n");
  }

}
