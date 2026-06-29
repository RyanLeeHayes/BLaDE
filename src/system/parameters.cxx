#include <omp.h>
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
  char rawLine[MAXLENGTHSTRING]; // DrudeIns - provenance marker for Drude PR.
  char token[MAXLENGTHSTRING];
  bool charmmStreamMode=false; // DrudeIns - provenance marker for Drude PR.
  bool inParameterBlock=true; // DrudeIns - provenance marker for Drude PR.
  bool skipCharmmBlock=false; // DrudeIns - provenance marker for Drude PR.

  while (fgets(line, MAXLENGTHSTRING, fp) != NULL) {
  // DrudeDel - original BLaDE line(s) removed or replaced for Drude support in this hunk.
    strncpy(rawLine,line,MAXLENGTHSTRING); // DrudeIns - provenance marker for Drude PR.
    rawLine[MAXLENGTHSTRING-1]='\0'; // DrudeIns - provenance marker for Drude PR.
    io_nexta(line,token); // DrudeIns - provenance marker for Drude PR.
    std::string TOKEN=io_uppers(token); // DrudeIns - provenance marker for Drude PR.
 // DrudeIns - provenance marker for Drude PR.
    if (skipCharmmBlock) { // DrudeIns - provenance marker for Drude PR.
      if (TOKEN=="END") { // DrudeIns - provenance marker for Drude PR.
        skipCharmmBlock=false; // DrudeIns - provenance marker for Drude PR.
        inParameterBlock=false; // DrudeIns - provenance marker for Drude PR.
      } // DrudeIns - provenance marker for Drude PR.
      continue;
    }
  // DrudeDel - original BLaDE line(s) removed or replaced for Drude support in this hunk.

    if (strcmp(token,"")==0) { // DrudeIns - provenance marker for Drude PR.
      ; // DrudeIns - provenance marker for Drude PR.
  // DrudeDel - original BLaDE line(s) removed or replaced for Drude support in this hunk.
    } else if (token[0]=='*') { // DrudeIns - provenance marker for Drude PR.
      fprintf(stdout,"TITLE> %s",rawLine); // DrudeIns - provenance marker for Drude PR.
    } else if (TOKEN=="READ") { // DrudeIns - provenance marker for Drude PR.
      std::string block=io_uppers(io_nexts(line)); // DrudeIns - provenance marker for Drude PR.
      if (block.compare(0,4,"PARA")==0) { // DrudeIns - provenance marker for Drude PR.
        charmmStreamMode=true; // DrudeIns - provenance marker for Drude PR.
        inParameterBlock=true; // DrudeIns - provenance marker for Drude PR.
      } else if (block=="RTF" || block.compare(0,4,"TOPO")==0) { // DrudeIns - provenance marker for Drude PR.
        charmmStreamMode=true; // DrudeIns - provenance marker for Drude PR.
        inParameterBlock=false; // DrudeIns - provenance marker for Drude PR.
        skipCharmmBlock=true; // DrudeIns - provenance marker for Drude PR.
      } else { // DrudeIns - provenance marker for Drude PR.
        fprintf(stdout,"Unparsed line> %s %s",token,line); // DrudeIns - provenance marker for Drude PR.
      } // DrudeIns - provenance marker for Drude PR.
    } else if (charmmStreamMode && !inParameterBlock) { // DrudeIns - provenance marker for Drude PR.
      ; // DrudeIns - provenance marker for Drude PR.
    } else if (TOKEN.compare(0,4,"ATOM")==0) { // DrudeIns - provenance marker for Drude PR.
      add_parameter_atoms(fp); // DrudeIns - provenance marker for Drude PR.
  // DrudeDel - original BLaDE line(s) removed or replaced for Drude support in this hunk.
    } else if (TOKEN.compare(0,4,"BOND")==0) { // DrudeIns - provenance marker for Drude PR.
      add_parameter_bonds(fp);
  // DrudeDel - original BLaDE line(s) removed or replaced for Drude support in this hunk.
    } else if (TOKEN.compare(0,4,"ANGL")==0) { // DrudeIns - provenance marker for Drude PR.
      add_parameter_angles(fp); // DrudeIns - provenance marker for Drude PR.
  // DrudeDel - original BLaDE line(s) removed or replaced for Drude support in this hunk.
    } else if (TOKEN.compare(0,4,"DIHE")==0) { // DrudeIns - provenance marker for Drude PR.
      add_parameter_dihes(fp);
  // DrudeDel - original BLaDE line(s) removed or replaced for Drude support in this hunk.
    } else if (TOKEN.compare(0,4,"IMPR")==0) { // DrudeIns - provenance marker for Drude PR.
      add_parameter_imprs(fp); // DrudeIns - provenance marker for Drude PR.
  // DrudeDel - original BLaDE line(s) removed or replaced for Drude support in this hunk.
    } else if (TOKEN.compare(0,4,"CMAP")==0) { // DrudeIns - provenance marker for Drude PR.
      add_parameter_cmaps(fp); // DrudeIns - provenance marker for Drude PR.
  // DrudeDel - original BLaDE line(s) removed or replaced for Drude support in this hunk.
    } else if (TOKEN.compare(0,4,"NONB")==0) { // DrudeIns - provenance marker for Drude PR.
      add_parameter_nbonds(line,fp); // DrudeIns - provenance marker for Drude PR.
  // DrudeDel - original BLaDE line(s) removed or replaced for Drude support in this hunk.
    } else if (TOKEN.compare(0,4,"NBFI")==0) { // DrudeIns - provenance marker for Drude PR.
      add_parameter_nbfixs(fp);
  // DrudeDel - original BLaDE line(s) removed or replaced for Drude support in this hunk.
    } else if (TOKEN.compare(0,4,"THOL")==0 || TOKEN.compare(0,4,"NBTH")==0) { // DrudeIns - provenance marker for Drude PR.
      add_parameter_tholes(line,fp); // DrudeIns - provenance marker for Drude PR.
    } else if (TOKEN=="END") { // DrudeIns - provenance marker for Drude PR.
      if (charmmStreamMode) { // DrudeIns - provenance marker for Drude PR.
        inParameterBlock=false; // DrudeIns - provenance marker for Drude PR.
        continue; // DrudeIns - provenance marker for Drude PR.
      } // DrudeIns - provenance marker for Drude PR.
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
      atomType.push_back(name);
      atomMass[name]=io_nextf(line);
      atomTypeCount++;
    } else {
      // This is part of next section. Back up and return.
      fsetpos(fp,&fp_pos);
      return; // DrudeIns - provenance marker for Drude PR.
    }

    fgetpos(fp,&fp_pos);
  }
}

void Parameters::add_parameter_bonds(FILE *fp) // DrudeIns - provenance marker for Drude PR.
{
  fpos_t fp_pos;
  char line[MAXLENGTHSTRING]; // DrudeIns - provenance marker for Drude PR.
  // DrudeDel - original BLaDE line(s) removed or replaced for Drude support in this hunk.
  std::string iname,INAME; // DrudeIns - provenance marker for Drude PR.
  TypeName2 name;
  struct BondParameter bp;

  fgetpos(fp,&fp_pos);
  while (fgets(line, MAXLENGTHSTRING, fp) != NULL) {
    iname=io_nexts(line);
    INAME=io_uppers(iname); // DrudeIns - provenance marker for Drude PR.
    if (strcmp(iname.c_str(),"")==0) {
      ;
  // DrudeDel - original BLaDE line(s) removed or replaced for Drude support in this hunk.
    } else if (knownTokens.count(INAME.substr(0,4))==0) { // DrudeIns - provenance marker for Drude PR.
      name.t[0]=check_type_name(iname,"BONDS");
      name.t[1]=check_type_name(io_nexts(line),"BONDS");
      bp.kb=(2.0*KCAL_MOL/(ANGSTROM*ANGSTROM))*io_nextf(line);
      bp.b0=ANGSTROM*io_nextf(line);
      bondParameter[name]=bp;
    } else {
      // This is part of next section. Back up and return. // DrudeIns - provenance marker for Drude PR.
      fsetpos(fp,&fp_pos);
      return;
    }

    fgetpos(fp,&fp_pos);
  }
} // DrudeIns - provenance marker for Drude PR.

void Parameters::add_parameter_angles(FILE *fp)
{ // DrudeIns - provenance marker for Drude PR.
  fpos_t fp_pos;
  char line[MAXLENGTHSTRING];
  // DrudeDel - original BLaDE line(s) removed or replaced for Drude support in this hunk.
  std::string iname,INAME; // DrudeIns - provenance marker for Drude PR.
  TypeName3 name;
  struct AngleParameter ap;

  fgetpos(fp,&fp_pos);
  while (fgets(line, MAXLENGTHSTRING, fp) != NULL) {
    iname=io_nexts(line);
    INAME=io_uppers(iname); // DrudeIns - provenance marker for Drude PR.
    if (strcmp(iname.c_str(),"")==0) {
      ;
  // DrudeDel - original BLaDE line(s) removed or replaced for Drude support in this hunk.
    } else if (knownTokens.count(INAME.substr(0,4))==0) { // DrudeIns - provenance marker for Drude PR.
      name.t[0]=check_type_name(iname,"ANGLES");
      name.t[1]=check_type_name(io_nexts(line),"ANGLES");
      name.t[2]=check_type_name(io_nexts(line),"ANGLES");
      ap.kangle=(2.0*KCAL_MOL)*io_nextf(line);
      ap.angle0=DEGREES*io_nextf(line);
      ap.kureyb=(2.0*KCAL_MOL/(ANGSTROM*ANGSTROM))*io_nextf(line,0);
      ap.ureyb0=ANGSTROM*io_nextf(line,0);
      angleParameter[name]=ap;
    // DrudeDel - original BLaDE fallback branch is preserved, but this hunk now uses upper-case token matching.
    } else { // DrudeIns - provenance marker for Drude PR.
      // This is part of next section. Back up and return.
      fsetpos(fp,&fp_pos);
      return;
    }

    fgetpos(fp,&fp_pos);
  }
}
 // DrudeIns - provenance marker for Drude PR.
// V(dihedral) = Kdih(1 + cos(ndih(chi) - dih0))
void Parameters::add_parameter_dihes(FILE *fp)
{ // DrudeIns - provenance marker for Drude PR.
  fpos_t fp_pos;
  char line[MAXLENGTHSTRING];
  // DrudeDel - original BLaDE line(s) removed or replaced for Drude support in this hunk.
  std::string iname,INAME; // DrudeIns - provenance marker for Drude PR.
  TypeName4 name;
  struct DiheParameter dp;
  std::vector<struct DiheParameter> dpv;
  int diheTerms;

  fgetpos(fp,&fp_pos);
  while (fgets(line, MAXLENGTHSTRING, fp) != NULL) {
    iname=io_nexts(line);
    INAME=io_uppers(iname); // DrudeIns - provenance marker for Drude PR.
    if (strcmp(iname.c_str(),"")==0) {
      ;
  // DrudeDel - original BLaDE line(s) removed or replaced for Drude support in this hunk.
    } else if (knownTokens.count(INAME.substr(0,4))==0) { // DrudeIns - provenance marker for Drude PR.
      name.t[0]=check_type_name(iname,"DIHEDRALS");
      name.t[1]=check_type_name(io_nexts(line),"DIHEDRALS");
      name.t[2]=check_type_name(io_nexts(line),"DIHEDRALS");
      name.t[3]=check_type_name(io_nexts(line),"DIHEDRALS");
      dp.kdih=KCAL_MOL*io_nextf(line);
      dp.ndih=io_nexti(line);
      dp.dih0=DEGREES*io_nextf(line);
      if (diheParameter.count(name)==1) {
        diheParameter[name].push_back(dp);
      } else {
        dpv.clear();
        dpv.push_back(dp);
        diheParameter[name]=dpv;
      } // DrudeIns - provenance marker for Drude PR.
      diheTerms=diheParameter[name].size();
      maxDiheTerms=((maxDiheTerms<diheTerms)?diheTerms:maxDiheTerms);
    } else {
      // This is part of next section. Back up and return.
      fsetpos(fp,&fp_pos);
      return;
    } // DrudeIns - provenance marker for Drude PR.

    fgetpos(fp,&fp_pos);
  } // DrudeIns - provenance marker for Drude PR.
}

void Parameters::add_parameter_imprs(FILE *fp)
{
  fpos_t fp_pos;
  char line[MAXLENGTHSTRING];
  // DrudeDel - original BLaDE line(s) removed or replaced for Drude support in this hunk.
  std::string iname,INAME; // DrudeIns - provenance marker for Drude PR.
  TypeName4 name;
  struct ImprParameter ip;

  fgetpos(fp,&fp_pos);
  while (fgets(line, MAXLENGTHSTRING, fp) != NULL) {
    iname=io_nexts(line);
    INAME=io_uppers(iname); // DrudeIns - provenance marker for Drude PR.
    if (strcmp(iname.c_str(),"")==0) {
      ;
  // DrudeDel - original BLaDE line(s) removed or replaced for Drude support in this hunk.
    } else if (knownTokens.count(INAME.substr(0,4))==0) { // DrudeIns - provenance marker for Drude PR.
      name.t[0]=check_type_name(iname,"IMPROPERS");
      name.t[1]=check_type_name(io_nexts(line),"IMPROPERS");
      name.t[2]=check_type_name(io_nexts(line),"IMPROPERS");
      name.t[3]=check_type_name(io_nexts(line),"IMPROPERS");
      ip.kimp=KCAL_MOL*io_nextf(line);
      ip.nimp=io_nexti(line);
      if (ip.nimp==0) {
        ip.kimp*=2;
      // DrudeDel - original improper periodicity branch is preserved after upper-case token matching.
      } else if (ip.nimp<0) { // DrudeIns - provenance marker for Drude PR.
        fatal(__FILE__,__LINE__,"Error: Improper periodicity in parameter file is less than 0.\n");
      }
      ip.imp0=DEGREES*io_nextf(line);
      imprParameter[name]=ip;
    } else {
      // This is part of next section. Back up and return.
      fsetpos(fp,&fp_pos);
      // DrudeDel - original section-return branch is preserved after upper-case token matching.
      return; // DrudeIns - provenance marker for Drude PR.
    }

    fgetpos(fp,&fp_pos); // DrudeIns - provenance marker for Drude PR.
  }
}

void Parameters::add_parameter_cmaps(FILE *fp)
{
  fpos_t fp_pos;
  char line[MAXLENGTHSTRING];
  // DrudeDel - original BLaDE line(s) removed or replaced for Drude support in this hunk.
  std::string iname,INAME; // DrudeIns - provenance marker for Drude PR.
  TypeName8O name;
  CmapParameter cp;
  int i,j;

  fgetpos(fp,&fp_pos);
  while (fgets(line, MAXLENGTHSTRING, fp) != NULL) {
    iname=io_nexts(line);
    INAME=io_uppers(iname); // DrudeIns - provenance marker for Drude PR.
    if (strcmp(iname.c_str(),"")==0) {
      ;
  // DrudeDel - original BLaDE line(s) removed or replaced for Drude support in this hunk.
    } else if (knownTokens.count(INAME.substr(0,4))==0) { // DrudeIns - provenance marker for Drude PR.
      name.t[0]=check_type_name(iname,"CMAPS");
      name.t[1]=check_type_name(io_nexts(line),"CMAPS");
      name.t[2]=check_type_name(io_nexts(line),"CMAPS");
      name.t[3]=check_type_name(io_nexts(line),"CMAPS");
      name.t[4]=check_type_name(io_nexts(line),"CMAPS");
      name.t[5]=check_type_name(io_nexts(line),"CMAPS");
      name.t[6]=check_type_name(io_nexts(line),"CMAPS");
      name.t[7]=check_type_name(io_nexts(line),"CMAPS");
      cp.ngrid=io_nexti(line);
      if (cp.ngrid>60) {
        fatal(__FILE__,__LINE__,"CMAP grid is greater than 60 points per 360 degrees (%d). Have you really thought about how much memory that will take?\n",cp.ngrid);
      }
      cp.kcmap=(real*)calloc(cp.ngrid*cp.ngrid,sizeof(real));
    fprintf(stdout,"allocating kcmap=%p\n",cp.kcmap);

      for (i=0; i<cp.ngrid; i++) {
        for (j=0; j<cp.ngrid; j++) {
          // Persistent search
          cp.kcmap[cp.ngrid*i+j]=KCAL_MOL*io_nextf(line,fp,"cmap matrix element");
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
  std::string iname,INAME;
  struct NbondParameter np;
  double e14fac=1;
  int combine=0; // 0 normal, 1 geometric

  fgetpos(fp,&fp_pos);
  while (fgets(line, MAXLENGTHSTRING, fp) != NULL) {
    iname=io_nexts(line);
    INAME=io_uppers(iname);
    // DrudeDel - original NONBONDED parser branch is preserved, but token matching now uses INAME.
    if (strcmp(iname.c_str(),"")==0) { // DrudeIns - provenance marker for Drude PR.
      ;
// Other acceptable tokens
//      cutnb  14.0 ctofnb 12.0 ctonnb 10.0 eps 1.0 e14fac 1.0 wmin 1.5 
    } else if (strcmp(INAME.c_str(),"NONBONDED")==0 ||
               strcmp(INAME.c_str(),"CUTNB")==0 ||
               strcmp(INAME.c_str(),"CTOFNB")==0 ||
               strcmp(INAME.c_str(),"CTONNB")==0 ||
               strcmp(INAME.c_str(),"EPS")==0 ||
               strcmp(INAME.c_str(),"E14FAC")==0 ||
               strcmp(INAME.c_str(),"WMIN")==0 ||
               strcmp(INAME.c_str(),"GEOM")==0) {
      while (strcmp(INAME.c_str(),"")!=0) {
        if (strcmp(INAME.c_str(),"E14FAC")==0) {
          e14fac=io_nextf(line);
        } else if (strcmp(INAME.c_str(),"GEOM")==0) {
          combine=1;
        }
        iname=io_nexts(line);
        INAME=io_uppers(iname);
      }
  // DrudeDel - original BLaDE line(s) removed or replaced for Drude support in this hunk.
    } else if (knownTokens.count(INAME.substr(0,4))==0) { // DrudeIns - provenance marker for Drude PR.
      check_type_name(iname,"NONBONDEDS");
      io_nextf(line);
      np.eps=io_nextf(line);
      np.sig=io_nextf(line);
      io_nextf(line,0);
      np.eps14=io_nextf(line,np.eps);
      np.sig14=io_nextf(line,np.sig); // DrudeIns - provenance marker for Drude PR.
      np.eps*=-KCAL_MOL;
      np.sig*=ANGSTROM;
      np.eps14*=-KCAL_MOL;
      np.sig14*=ANGSTROM;
      np.e14fac=e14fac;
      np.combine=combine;
      nbondParameter[iname]=np; // DrudeIns - provenance marker for Drude PR.
    } else {
      // This is part of next section. Back up and return.
      fsetpos(fp,&fp_pos); // DrudeIns - provenance marker for Drude PR.
      return;
    }

    fgetpos(fp,&fp_pos);
  }
}

void Parameters::add_parameter_nbfixs(FILE *fp)
{
  fpos_t fp_pos;
  char line[MAXLENGTHSTRING];
  // DrudeDel - original BLaDE line(s) removed or replaced for Drude support in this hunk.
  std::string iname,INAME; // DrudeIns - provenance marker for Drude PR.
  TypeName2 name;
  struct NbondParameter np;

  fgetpos(fp,&fp_pos);
  while (fgets(line, MAXLENGTHSTRING, fp) != NULL) {
    iname=io_nexts(line);
    INAME=io_uppers(iname); // DrudeIns - provenance marker for Drude PR.
    if (strcmp(iname.c_str(),"")==0) {
      ;
  // DrudeDel - original BLaDE line(s) removed or replaced for Drude support in this hunk.
    } else if (knownTokens.count(INAME.substr(0,4))==0) { // DrudeIns - provenance marker for Drude PR.
      name.t[0]=check_type_name(iname,"NBFIX"); // DrudeIns - provenance marker for Drude PR.
      name.t[1]=check_type_name(io_nexts(line),"NBFIX"); // DrudeIns - provenance marker for Drude PR.
      np.eps=io_nextf(line); // DrudeIns - provenance marker for Drude PR.
      np.sig=io_nextf(line); // DrudeIns - provenance marker for Drude PR.
      np.eps14=io_nextf(line,np.eps);
      np.sig14=io_nextf(line,np.sig); // DrudeIns - provenance marker for Drude PR.
      np.eps*=-KCAL_MOL; // DrudeIns - provenance marker for Drude PR.
      np.sig*=ANGSTROM; // DrudeIns - provenance marker for Drude PR.
      np.eps14*=-KCAL_MOL; // DrudeIns - provenance marker for Drude PR.
      np.sig14*=ANGSTROM; // DrudeIns - provenance marker for Drude PR.
      np.e14fac=1; // unused // DrudeIns - provenance marker for Drude PR.
      np.combine=0; // unused // DrudeIns - provenance marker for Drude PR.
      nbfixParameter[name]=np; // DrudeIns - provenance marker for Drude PR.
    } else { // DrudeIns - provenance marker for Drude PR.
      // This is part of next section. Back up and return. // DrudeIns - provenance marker for Drude PR.
      fsetpos(fp,&fp_pos); // DrudeIns - provenance marker for Drude PR.
      return; // DrudeIns - provenance marker for Drude PR.
    } // DrudeIns - provenance marker for Drude PR.
 // DrudeIns - provenance marker for Drude PR.
    fgetpos(fp,&fp_pos); // DrudeIns - provenance marker for Drude PR.
  } // DrudeIns - provenance marker for Drude PR.
} // DrudeIns - provenance marker for Drude PR.
 // DrudeIns - provenance marker for Drude PR.
void Parameters::add_parameter_tholes(char *line,FILE *fp) // DrudeIns - provenance marker for Drude PR.
{ // DrudeIns - provenance marker for Drude PR.
  fpos_t fp_pos; // DrudeIns - provenance marker for Drude PR.
  std::string iname,INAME; // DrudeIns - provenance marker for Drude PR.
  TypeName2 name; // DrudeIns - provenance marker for Drude PR.

  fgetpos(fp,&fp_pos); // DrudeIns - provenance marker for Drude PR.
  while (fgets(line, MAXLENGTHSTRING, fp) != NULL) { // DrudeIns - provenance marker for Drude PR.
    iname=io_nexts(line); // DrudeIns - provenance marker for Drude PR.
    INAME=io_uppers(iname); // DrudeIns - provenance marker for Drude PR.
    if (strcmp(iname.c_str(),"")==0) { // DrudeIns - provenance marker for Drude PR.
      ; // DrudeIns - provenance marker for Drude PR.
    } else if (knownTokens.count(INAME.substr(0,4))==0) { // DrudeIns - provenance marker for Drude PR.
      name.t[0]=check_type_name(iname,"THOLE"); // DrudeIns - provenance marker for Drude PR.
      name.t[1]=check_type_name(io_nexts(line),"THOLE"); // DrudeIns - provenance marker for Drude PR.
      tholePairParameter[name]=io_nextf(line); // DrudeIns - provenance marker for Drude PR.
    } else {
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

  for (std::map<std::string,real>::iterator ii=atomMass.begin(); ii!=atomMass.end(); ii++) {
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

  for (std::map<TypeName4,std::vector<struct DiheParameter> >::iterator ii=diheParameter.begin(); ii!=diheParameter.end(); ii++) {
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
  // DrudeDel - original dump ordering is preserved, with THOLE parameter printing added below.
  fprintf(stdout,"%s maxDiheTerms=%d\n",tag,maxDiheTerms); // DrudeIns - provenance marker for Drude PR.
  fprintf(stdout,"%s\n",tag); // DrudeIns - provenance marker for Drude PR.
 // DrudeIns - provenance marker for Drude PR.
  for (std::map<TypeName4,struct ImprParameter>::iterator ii=imprParameter.begin(); ii!=imprParameter.end(); ii++) { // DrudeIns - provenance marker for Drude PR.
    TypeName4 name=ii->first; // DrudeIns - provenance marker for Drude PR.
    struct ImprParameter ip=ii->second;
    fprintf(stdout,"%s imprParameter[%6s,%6s,%6s,%6s]={kimp=%g nimp=%d imp0=%g}\n",tag,name.t[0].c_str(),name.t[1].c_str(),name.t[2].c_str(),name.t[3].c_str(),ip.kimp,ip.nimp,ip.imp0);
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

  for (std::map<TypeName2,real>::iterator ii=tholePairParameter.begin(); ii!=tholePairParameter.end(); ii++) { // DrudeIns - provenance marker for Drude PR.
    TypeName2 name=ii->first; // DrudeIns - provenance marker for Drude PR.
    fprintf(stdout,"%s tholePairParameter[%6s,%6s]={thole=%g}\n",tag,name.t[0].c_str(),name.t[1].c_str(),ii->second); // DrudeIns - provenance marker for Drude PR.
  } // DrudeIns - provenance marker for Drude PR.
  fprintf(stdout,"%s\n",tag); // DrudeIns - provenance marker for Drude PR.

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



void blade_init_parameters(System *system)
{
  system+=omp_get_thread_num();
  if (system->parameters) {
    delete(system->parameters);
  }
  system->parameters=new Parameters();
}

void blade_dest_parameters(System *system)
{
  system+=omp_get_thread_num();
  if (system->parameters) {
    delete(system->parameters);
  }
  system->parameters=NULL;
}

void blade_add_parameter_atoms(System *system,const char *name,double mass)
{
  system+=omp_get_thread_num();
  system->parameters->atomTypeMap[name]=system->parameters->atomTypeCount;
  system->parameters->atomType.push_back(name);
  system->parameters->atomMass[name]=mass;
  system->parameters->atomTypeCount++;
}

void blade_add_parameter_bonds(System *system,const char *t1,const char *t2,double kb,double b0)
{
  TypeName2 name;
  struct BondParameter bp;

  name.t[0]=t1;
  name.t[1]=t2;
  bp.kb=(2.0*KCAL_MOL/(ANGSTROM*ANGSTROM))*kb;
  bp.b0=ANGSTROM*b0;
  system+=omp_get_thread_num();
  system->parameters->bondParameter[name]=bp;
}

void blade_add_parameter_angles(System *system,const char *t1,const char *t2,const char *t3,double kangle,double angle0,double kureyb,double ureyb0)
{
  TypeName3 name;
  struct AngleParameter ap;

  name.t[0]=t1;
  name.t[1]=t2;
  name.t[2]=t3;
  ap.kangle=(2.0*KCAL_MOL)*kangle;
  ap.angle0=DEGREES*angle0;
  ap.kureyb=(2.0*KCAL_MOL/(ANGSTROM*ANGSTROM))*kureyb;
  ap.ureyb0=ANGSTROM*ureyb0;
  system+=omp_get_thread_num();
  system->parameters->angleParameter[name]=ap;
}

// V(dihedral) = Kdih(1 + cos(ndih(chi) - dih0))
void blade_add_parameter_dihes(System *system,const char *t1,const char *t2,const char *t3,const char *t4,double kdih,int ndih,double dih0)
{
  TypeName4 name;
  struct DiheParameter dp;
  std::vector<struct DiheParameter> dpv;
  int diheTerms;

  name.t[0]=t1;
  name.t[1]=t2;
  name.t[2]=t3;
  name.t[3]=t4;
  dp.kdih=KCAL_MOL*kdih;
  dp.ndih=ndih;
  dp.dih0=DEGREES*dih0;
  system+=omp_get_thread_num();
  if (system->parameters->diheParameter.count(name)==1) {
    system->parameters->diheParameter[name].push_back(dp);
  } else {
    dpv.clear();
    dpv.push_back(dp);
    system->parameters->diheParameter[name]=dpv;
  }
  diheTerms=system->parameters->diheParameter[name].size();
  system->parameters->maxDiheTerms=((system->parameters->maxDiheTerms<diheTerms)?diheTerms:system->parameters->maxDiheTerms);
}

void blade_add_parameter_imprs(System *system,const char *t1,const char *t2,const char *t3,const char *t4,double kimp,int nimp,double imp0)
{
  TypeName4 name;
  struct ImprParameter ip;

  name.t[0]=t1;
  name.t[1]=t2;
  name.t[2]=t3;
  name.t[3]=t4;
  ip.kimp=KCAL_MOL*kimp;
  ip.nimp=nimp;
  if (ip.nimp==0) {
    ip.kimp*=2;
  } else if (ip.nimp<0) {
    fatal(__FILE__,__LINE__,"Error: Improper periodicity in parameter file is less than 0.\n");
  }
  ip.imp0=DEGREES*imp0;
  system+=omp_get_thread_num();
  system->parameters->imprParameter[name]=ip;
}

void blade_add_parameter_cmaps(System *system,
  const char *t1,const char *t2,const char *t3,const char *t4,
  const char *t5,const char *t6,const char *t7,const char *t8,
  int ngrid)
{
  TypeName8O name;
  CmapParameter cp;

  name.t[0]=t1;
  name.t[1]=t2;
  name.t[2]=t3;
  name.t[3]=t4;
  name.t[4]=t5;
  name.t[5]=t6;
  name.t[6]=t7;
  name.t[7]=t8;
  cp.ngrid=ngrid;
  if (cp.ngrid>60) {
    fatal(__FILE__,__LINE__,"CMAP grid is greater than 60 points per 360 degrees (%d). Have you really thought about how much memory that will take?\n",cp.ngrid);
  }
  system+=omp_get_thread_num();
  cp.kcmap=(real*)calloc(cp.ngrid*cp.ngrid,sizeof(real));
  fprintf(stdout,"allocating kcmap=%p\n",cp.kcmap);
  system->parameters->cmapParameter[name]=cp;
}

void blade_add_parameter_cmaps_fill(System *system,
  const char *t1,const char *t2,const char *t3,const char *t4,
  const char *t5,const char *t6,const char *t7,const char *t8,
  int i,int j,double kcmapij)
{
  TypeName8O name;

  name.t[0]=t1;
  name.t[1]=t2;
  name.t[2]=t3;
  name.t[3]=t4;
  name.t[4]=t5;
  name.t[5]=t6;
  name.t[6]=t7;
  name.t[7]=t8;
  system+=omp_get_thread_num();
  int ngrid=system->parameters->cmapParameter[name].ngrid;
  if (i-1<ngrid) {
    if (j-1<ngrid) {
      system->parameters->cmapParameter[name].kcmap[ngrid*(i-1)+(j-1)]=KCAL_MOL*kcmapij;
    }
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
void blade_add_parameter_nbonds(System *system,const char *t1,double eps,double sig,double eps14,double sig14,double e14fac,int combine)
{
  std::string name;
  struct NbondParameter np;

  name=t1;
  np.eps=eps;
  np.sig=sig;
  np.eps14=eps14;
  np.sig14=sig14;
  np.eps*=-KCAL_MOL;
  np.sig*=ANGSTROM;
  np.eps14*=-KCAL_MOL;
  np.sig14*=ANGSTROM;
  np.e14fac=e14fac;
  np.combine=combine; // combination rule, 0 is normal charmm combination rule, 1 is geometric combination rule
  system+=omp_get_thread_num();
  system->parameters->nbondParameter[name]=np;
}

void blade_add_parameter_nbfixs(System *system,const char *t1,const char *t2,double eps,double sig,double eps14,double sig14)
{
  TypeName2 name;
  struct NbondParameter np;

  name.t[0]=t1;
  name.t[1]=t2;
  np.eps=eps;
  np.sig=sig;
  np.eps14=eps14;
  np.sig14=sig14;
  np.eps*=-KCAL_MOL;
  np.sig*=ANGSTROM;
  np.eps14*=-KCAL_MOL;
  np.sig14*=ANGSTROM;
  system+=omp_get_thread_num();
  system->parameters->nbfixParameter[name]=np;
}
