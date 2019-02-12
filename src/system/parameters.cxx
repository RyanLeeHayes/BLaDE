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

bool operator==(const TypeName2& a,const TypeName2& b)
{
  return (a.t[0]==b.t[0] && a.t[1]==b.t[1])||(a.t[0]==b.t[1] && a.t[1]==b.t[0]);
}
bool operator<(const TypeName2& a,const TypeName2& b)
{
  int i,aReverse,bReverse;
  int N=2;
  for (i=0; i<(N/2); i++) { // Decide whether to compare a in reverse order
    aReverse=((a.t[i]<a.t[N-1-i])?0:1);
    if (a.t[i]!=a.t[N-1-i]) break;
  }
  for (i=0; i<(N/2); i++) { // Decide whether to compare b in reverse order
    bReverse=((b.t[i]<b.t[N-1-i])?0:1);
    if (b.t[i]!=b.t[N-1-i]) break;
  }
  for (i=0; i<N; i++) {
    if (a.t[i+aReverse*(N-1-2*i)]==b.t[i+bReverse*(N-1-2*i)]) continue;
    return (a.t[i+aReverse*(N-1-2*i)]<b.t[i+bReverse*(N-1-2*i)]);
  }
  return false;
}

bool operator==(const TypeName3& a,const TypeName3& b)
{
  return (a.t[0]==b.t[0] && a.t[1]==b.t[1] && a.t[2]==b.t[2])||(a.t[0]==b.t[2] && a.t[1]==b.t[1] && a.t[2]==b.t[0]);
}
bool operator<(const TypeName3& a,const TypeName3& b)
{
  int i,aReverse,bReverse;
  int N=3;
  for (i=0; i<(N/2); i++) { // Decide whether to compare a in reverse order
    aReverse=((a.t[i]<a.t[N-1-i])?0:1);
    if (a.t[i]!=a.t[N-1-i]) break;
  }
  for (i=0; i<(N/2); i++) { // Decide whether to compare b in reverse order
    bReverse=((b.t[i]<b.t[N-1-i])?0:1);
    if (b.t[i]!=b.t[N-1-i]) break;
  }
  for (i=0; i<N; i++) {
    if (a.t[i+aReverse*(N-1-2*i)]==b.t[i+bReverse*(N-1-2*i)]) continue;
    return (a.t[i+aReverse*(N-1-2*i)]<b.t[i+bReverse*(N-1-2*i)]);
  }
  return false;
}

bool operator==(const TypeName4& a,const TypeName4& b)
{
  return (a.t[0]==b.t[0] && a.t[1]==b.t[1] && a.t[2]==b.t[2] && a.t[3]==b.t[3])||(a.t[0]==b.t[3] && a.t[1]==b.t[2] && a.t[2]==b.t[1] && a.t[3]==b.t[0]);
}
bool operator<(const TypeName4& a,const TypeName4& b)
{
  int i,aReverse,bReverse;
  int N=4;
  for (i=0; i<(N/2); i++) { // Decide whether to compare a in reverse order
    aReverse=((a.t[i]<a.t[N-1-i])?0:1);
    if (a.t[i]!=a.t[N-1-i]) break;
  }
  for (i=0; i<(N/2); i++) { // Decide whether to compare b in reverse order
    bReverse=((b.t[i]<b.t[N-1-i])?0:1);
    if (b.t[i]!=b.t[N-1-i]) break;
  }
  for (i=0; i<N; i++) {
    if (a.t[i+aReverse*(N-1-2*i)]==b.t[i+bReverse*(N-1-2*i)]) continue;
    return (a.t[i+aReverse*(N-1-2*i)]<b.t[i+bReverse*(N-1-2*i)]);
  }
  return false;
}

bool operator==(const TypeName8O& a,const TypeName8O& b)
{
  return a.t[0]==b.t[0] && a.t[1]==b.t[1] && a.t[2]==b.t[2] && a.t[3]==b.t[3] && a.t[4]==b.t[4] && a.t[5]==b.t[5] && a.t[6]==b.t[6] && a.t[7]==b.t[7];
}
bool operator<(const TypeName8O& a,const TypeName8O& b)
{
  int i,aReverse,bReverse;
  int N=8;
  aReverse=0;
  bReverse=0;
  for (i=0; i<N; i++) {
    if (a.t[i+aReverse*(N-1-2*i)]==b.t[i+bReverse*(N-1-2*i)]) continue;
    return (a.t[i+aReverse*(N-1-2*i)]<b.t[i+bReverse*(N-1-2*i)]);
  }
  return false;
}

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
    fp=fpopen(token,"r");
    system->parameters->add_parameter_file(fp);
    fclose(fp);
  } else if (strcmp(token,"print")==0) {
    system->parameters->dump();
  } else {
    fatal(__FILE__,__LINE__,"Unrecognized parameters token: %s\n",token); // FIXIT add token name
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
      add_parameter_dihes(fp);
    } else if (strncmp(token,"IMPROPERS",4)==0) {
      add_parameter_imprs(fp);
    } else if (strncmp(token,"CMAP",4)==0) {
      add_parameter_cmaps(fp);
    } else if (strncmp(token,"NONBONDED",4)==0) {
      add_parameter_nbonds(line,fp);
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
  std::string name;

  fgetpos(fp,&fp_pos);
  while (fgets(line, MAXLENGTHSTRING, fp) != NULL) {
    io_nexta(line,token);
    if (strcmp(token,"")==0) {
      ;
    } else if (strcmp(token,"MASS")==0) {
      io_nexti(line); // Ignore index
      name=io_nexts(line);
      atomTypeMap[name]=atomTypeCount;
      atomType.emplace_back(name);
      atomMass[name]=io_nextf(line);
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
  TypeName2 name;
  struct BondParameter bp;

  fgetpos(fp,&fp_pos);
  while (fgets(line, MAXLENGTHSTRING, fp) != NULL) {
    iname=io_nexts(line);
    if (strcmp(iname.c_str(),"")==0) {
      ;
    } else if (knownTokens.count(iname.substr(0,4))==0) {
      name.t[0]=check_type_name(iname,"BONDS");
      name.t[1]=check_type_name(io_nexts(line),"BONDS");
      bp.kb=(2.0*KCAL_MOL/(ANGSTROM*ANGSTROM))*io_nextf(line);
      bp.b0=ANGSTROM*io_nextf(line);
      bondParameter[name]=bp;
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
  TypeName3 name;
  struct AngleParameter ap;

  fgetpos(fp,&fp_pos);
  while (fgets(line, MAXLENGTHSTRING, fp) != NULL) {
    iname=io_nexts(line);
    if (strcmp(iname.c_str(),"")==0) {
      ;
    } else if (knownTokens.count(iname.substr(0,4))==0) {
      name.t[0]=check_type_name(iname,"ANGLES");
      name.t[1]=check_type_name(io_nexts(line),"ANGLES");
      name.t[2]=check_type_name(io_nexts(line),"ANGLES");
      ap.kangle=(2.0*KCAL_MOL)*io_nextf(line);
      ap.angle0=DEGREES*io_nextf(line);
      ap.kureyb=(2.0*KCAL_MOL/(ANGSTROM*ANGSTROM))*io_nextf(line,0);
      ap.ureyb0=ANGSTROM*io_nextf(line,0);
      angleParameter[name]=ap;
    } else {
      // This is part of next section. Back up and return.
      fsetpos(fp,&fp_pos);
      return;
    }

    fgetpos(fp,&fp_pos);
  }
}

// V(dihedral) = Kdih(1 + cos(ndih(chi) - dih0))
void Parameters::add_parameter_dihes(FILE *fp)
{
  fpos_t fp_pos;
  char line[MAXLENGTHSTRING];
  std::string iname;
  TypeName4 name;
  struct DiheParameter dp;
  std::vector<struct DiheParameter> dpv;
  int diheTerms;

  fgetpos(fp,&fp_pos);
  while (fgets(line, MAXLENGTHSTRING, fp) != NULL) {
    iname=io_nexts(line);
    if (strcmp(iname.c_str(),"")==0) {
      ;
    } else if (knownTokens.count(iname.substr(0,4))==0) {
      name.t[0]=check_type_name(iname,"DIHEDRALS");
      name.t[1]=check_type_name(io_nexts(line),"DIHEDRALS");
      name.t[2]=check_type_name(io_nexts(line),"DIHEDRALS");
      name.t[3]=check_type_name(io_nexts(line),"DIHEDRALS");
      dp.kdih=KCAL_MOL*io_nextf(line);
      dp.ndih=io_nexti(line);
      dp.dih0=DEGREES*io_nextf(line);
      if (diheParameter.count(name)==1) {
        diheParameter[name].emplace_back(dp);
      } else {
        dpv.clear();
        dpv.emplace_back(dp);
        diheParameter[name]=dpv;
      }
      diheTerms=diheParameter[name].size();
      maxDiheTerms=((maxDiheTerms<diheTerms)?diheTerms:maxDiheTerms);
    } else {
      // This is part of next section. Back up and return.
      fsetpos(fp,&fp_pos);
      return;
    }

    fgetpos(fp,&fp_pos);
  }
}

void Parameters::add_parameter_imprs(FILE *fp)
{
  fpos_t fp_pos;
  char line[MAXLENGTHSTRING];
  std::string iname;
  TypeName4 name;
  struct ImprParameter ip;

  fgetpos(fp,&fp_pos);
  while (fgets(line, MAXLENGTHSTRING, fp) != NULL) {
    iname=io_nexts(line);
    if (strcmp(iname.c_str(),"")==0) {
      ;
    } else if (knownTokens.count(iname.substr(0,4))==0) {
      name.t[0]=check_type_name(iname,"IMPROPERS");
      name.t[1]=check_type_name(io_nexts(line),"IMPROPERS");
      name.t[2]=check_type_name(io_nexts(line),"IMPROPERS");
      name.t[3]=check_type_name(io_nexts(line),"IMPROPERS");
      ip.kimp=(2.0*KCAL_MOL)*io_nextf(line);
      io_nexts(line);
      ip.imp0=DEGREES*io_nextf(line);
      imprParameter[name]=ip;
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
  TypeName8O name;
  CmapParameter cp;
  int i,j;
  double k;
  char *escape;

  fgetpos(fp,&fp_pos);
  while (fgets(line, MAXLENGTHSTRING, fp) != NULL) {
    iname=io_nexts(line);
    if (strcmp(iname.c_str(),"")==0) {
      ;
    } else if (knownTokens.count(iname.substr(0,4))==0) {
      name.t[0]=check_type_name(iname,"CMAPS");
      name.t[1]=check_type_name(io_nexts(line),"CMAPS");
      name.t[2]=check_type_name(io_nexts(line),"CMAPS");
      name.t[3]=check_type_name(io_nexts(line),"CMAPS");
      name.t[4]=check_type_name(io_nexts(line),"CMAPS");
      name.t[5]=check_type_name(io_nexts(line),"CMAPS");
      name.t[6]=check_type_name(io_nexts(line),"CMAPS");
      name.t[7]=check_type_name(io_nexts(line),"CMAPS");
#warning "Assume cmap size of 24"
      cp.ngrid=io_nexti(line);
      if (cp.ngrid!=24) {
        fatal(__FILE__,__LINE__,"Error: lazy coding, programmer assumed CMAP size would be 24 when it was %d\n",cp.ngrid);
      }

      for (i=0; i<cp.ngrid; i++) {
        for (j=0; j<cp.ngrid; j++) {
          // Persistent search
          cp.kcmap[i][j]=KCAL_MOL*io_nextf(line,fp,"cmap matrix element");
        }
      }
      cmapParameter[name]=cp;
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
void Parameters::add_parameter_nbonds(char *line,FILE *fp)
{
  fpos_t fp_pos;
  // char line[MAXLENGTHSTRING];
  std::string iname;
  struct NbondParameter np;

  fgetpos(fp,&fp_pos);
  while (fgets(line, MAXLENGTHSTRING, fp) != NULL) {
    iname=io_nexts(line);
    if (strcmp(iname.c_str(),"")==0) {
      ;
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
    } else if (knownTokens.count(iname.substr(0,4))==0) {
      check_type_name(iname,"NONBONDEDS");
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
      nbondParameter[iname]=np;
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
  TypeName2 name;
  struct NbondParameter np;

  fgetpos(fp,&fp_pos);
  while (fgets(line, MAXLENGTHSTRING, fp) != NULL) {
    iname=io_nexts(line);
    if (strcmp(iname.c_str(),"")==0) {
      ;
    } else if (knownTokens.count(iname.substr(0,4))==0) {
      name.t[0]=check_type_name(iname,"NBFIX");
      name.t[1]=check_type_name(io_nexts(line),"NBFIX");
      np.eps=io_nextf(line);
      np.sig=io_nextf(line);
      np.eps14=io_nextf(line,np.eps);
      np.sig14=io_nextf(line,np.sig);
      np.eps*=KCAL_MOL;
      np.sig*=ANGSTROM;
      np.eps14*=KCAL_MOL;
      np.sig14*=ANGSTROM;
      nbfixParameter[name]=np;
    } else {
      // This is part of next section. Back up and return.
      fsetpos(fp,&fp_pos);
      return;
    }

    fgetpos(fp,&fp_pos);
  }
}


void Parameters::dump()
{
  int i;
  std::string key;
  char tag[]="PRINT PARAMETERS>";

  fprintf(stdout,"%s atomTypeCount=%d\n",tag,atomTypeCount);
  fprintf(stdout,"%s\n",tag);

  for (std::map<std::string,int>::iterator ii=atomTypeMap.begin(); ii!=atomTypeMap.end(); ii++) {
    fprintf(stdout,"%s atomTypeMap[%6s]=%5d\n",tag,ii->first.c_str(),ii->second);
  }
  fprintf(stdout,"%s\n",tag);

  for (i=0; i<atomTypeCount; i++) {
    fprintf(stdout,"%s atomType[%5d]=%6s\n",tag,i,atomType[i].c_str());
  }
  fprintf(stdout,"%s\n",tag);

  for (std::map<std::string,double>::iterator ii=atomMass.begin(); ii!=atomMass.end(); ii++) {
    fprintf(stdout,"%s atomMass[%6s]=%g\n",tag,ii->first.c_str(),ii->second);
  }
  fprintf(stdout,"%s\n",tag);

  for (std::map<TypeName2,struct BondParameter>::iterator ii=bondParameter.begin(); ii!=bondParameter.end(); ii++) {
    TypeName2 name=ii->first;
    struct BondParameter bp=ii->second;
    fprintf(stdout,"%s bondParameter[%6s,%6s]={kb=%g b0=%g}\n",tag,name.t[0].c_str(),name.t[1].c_str(),bp.kb,bp.b0);
  }
  fprintf(stdout,"%s\n",tag);

  for (std::map<TypeName3,struct AngleParameter>::iterator ii=angleParameter.begin(); ii!=angleParameter.end(); ii++) {
    TypeName3 name=ii->first;
    struct AngleParameter ap=ii->second;
    fprintf(stdout,"%s angleParameter[%6s,%6s,%6s]={kangle=%g angle0=%g kureyb=%g ureyb0=%g}\n",tag,name.t[0].c_str(),name.t[1].c_str(),name.t[2].c_str(),ap.kangle,ap.angle0,ap.kureyb,ap.ureyb0);
  }
  fprintf(stdout,"%s\n",tag);

  for (std::map<TypeName4,std::vector<struct DiheParameter>>::iterator ii=diheParameter.begin(); ii!=diheParameter.end(); ii++) {
    TypeName4 name=ii->first;
    std::vector<struct DiheParameter> dpv=ii->second;
    char varName[45]; // 42+\0
    sprintf(varName,"diheParameter[%6s,%6s,%6s,%6s]",name.t[0].c_str(),name.t[1].c_str(),name.t[2].c_str(),name.t[3].c_str());
    for (int j=0; j<dpv.size(); j++) {
      struct DiheParameter dp=dpv[j];
      fprintf(stdout,"%s %s={kdih=%g ndih=%d dih0=%g}\n",tag,varName,dp.kdih,dp.ndih,dp.dih0);
      sprintf(varName,"                                          ");
    }
  }
  fprintf(stdout,"%s\n",tag);
  fprintf(stdout,"%s maxDiheTerms=%d\n",tag,maxDiheTerms);
  fprintf(stdout,"%s\n",tag);

  for (std::map<TypeName4,struct ImprParameter>::iterator ii=imprParameter.begin(); ii!=imprParameter.end(); ii++) {
    TypeName4 name=ii->first;
    struct ImprParameter ip=ii->second;
    fprintf(stdout,"%s imprParameter[%6s,%6s,%6s,%6s]={kimp=%g imp0=%g}\n",tag,name.t[0].c_str(),name.t[1].c_str(),name.t[2].c_str(),name.t[3].c_str(),ip.kimp,ip.imp0);
  }
  fprintf(stdout,"%s\n",tag);

  for (std::map<std::string,struct NbondParameter>::iterator ii=nbondParameter.begin(); ii!=nbondParameter.end(); ii++) {
    std::string name=ii->first;
    struct NbondParameter np=ii->second;
    fprintf(stdout,"%s nbondParameter[%6s]={eps=%g sig=%g eps14=%g sig14=%g}\n",tag,name.c_str(),np.eps,np.sig,np.eps14,np.sig14);
  }
  fprintf(stdout,"%s\n",tag);

  for (std::map<TypeName2,struct NbondParameter>::iterator ii=nbfixParameter.begin(); ii!=nbfixParameter.end(); ii++) {
    TypeName2 name=ii->first;
    struct NbondParameter np=ii->second;
    fprintf(stdout,"%s nbfixParameter[%6s,%6s]={eps=%g sig=%g eps14=%g sig14=%g}\n",tag,name.t[0].c_str(),name.t[1].c_str(),np.eps,np.sig,np.eps14,np.sig14);
  }
  fprintf(stdout,"%s\n",tag);

  fprintf(stdout,"%s cmapParameter not printed\n",tag);
  fprintf(stdout,"%s\n",tag);

}

std::string Parameters::check_type_name(std::string type,const char *tag)
{ 
  if (atomTypeMap.count(type)==0) {
    fprintf(stdout,"Warning: atom type %s found in %s but not declared in ATOMS\n",type.c_str(),tag);
  }
  return type;
}

std::string Parameters::require_type_name(std::string type,const char *tag)
{ 
  if (atomTypeMap.count(type)==0) {
    fatal(__FILE__,__LINE__,"Atom type %s not found while %s\n",type.c_str(),tag);
  }
  return type;
}
