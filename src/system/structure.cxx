#include <omp.h>
#include <string.h>

#include "system/system.h"
#include "system/parameters.h"
#include "system/structure.h"
#include "system/selections.h"
#include "system/coordinates.h"
#include "system/potential.h"
#include "main/defines.h"
#include "io/io.h"



// Class constructors
Structure::Structure() {
  atomCount=0;

  bondCount=0;
  angleCount=0;
  diheCount=0;
  imprCount=0;
  cmapCount=0;

  virt2Count=0;
  virt3Count=0;

  shakeHbond=false;

  noeCount=0;
  noeList.clear();
  harmCount=0;
  harmList.clear();

  setup_parse_structure();
}

Structure::~Structure() {
}



// Parsing functions
void Structure::setup_parse_structure()
{
  parseStructure[""]=&Structure::error;
  helpStructure[""]="?> How did we get here?\n";
  parseStructure["reset"]=&Structure::reset;
  helpStructure["reset"]="?structure reset> This deletes the structure data structure.\n";
  parseStructure["file"]=&Structure::file;
  helpStructure["file"]="?structure file psf [filename]> This loads the system structure from the CHARMM PSF (protein structure file)\n";
  parseStructure["shake"]=&Structure::parse_shake;
  helpStructure["shake"]="?structure shake [hbond/none]> Turn hydrogen bond length constraints on or off.\n";
  parseStructure["noe"]=&Structure::parse_noe;
  helpStructure["noe"]="?structure noe [selection] [selection] [rmin] [kmin] [rmax] [kmax] [rpeak] [rswitch] [nswitch]> Apply a CHARMM-style NOE restraint between a pair of atoms\n";
  parseStructure["harmonic"]=&Structure::parse_harmonic;
  helpStructure["harmonic"]="?structure harmonic [selection] [mass|none] [k real] [n real]> Apply harmonic restraints of k*(x-x0)^n to each atom in selection. x0 is taken from the current coordinates read in by coordinates. For none, k has units of kcal/mol/A^n, for mass, k is multiplied by the mass, and has units of kcal/mol/A^n/amu. structure harmonic reset clears all restraints\n";
  parseStructure["print"]=&Structure::dump;
  helpStructure["print"]="?structure print> This prints selected contents of the structure data structure to standard out\n";
  parseStructure["help"]=&Structure::help;
  helpStructure["help"]="?structure help [directive]> Prints help on state directive, including a list of subdirectives. If a subdirective is listed, this prints help on that specific subdirective.\n";
}

void parse_structure(char *line,System *system)
{
  char token[MAXLENGTHSTRING];
  std::string name;

  if (!system->structure) {
    system->structure=new Structure();
  }

  io_nexta(line,token);
  name=token;
  if (system->structure->parseStructure.count(name)==0) name="";
  // So much for function pointers being elegant.
  // call the function pointed to by: system->structure->parseStructure[name]
  // within the object: system->structure
  // with arguments: (line,token,system)
  (system->structure->*(system->structure->parseStructure[name]))(line,token,system);
}

void Structure::help(char *line,char *token,System *system)
{
  char name[MAXLENGTHSTRING];
  io_nexta(line,name);
  if (name=="") {
    fprintf(stdout,"?structure > Available directives are:\n");
    for (std::map<std::string,std::string>::iterator ii=helpStructure.begin(); ii!=helpStructure.end(); ii++) {
      fprintf(stdout," %s",ii->first.c_str());
    }
    fprintf(stdout,"\n");
  } else if (helpStructure.count(name)==1) {
    fprintf(stdout,helpStructure[name].c_str());
  } else {
    error(line,name,system);
  }
}

void Structure::error(char *line,char *token,System *system)
{
  fatal(__FILE__,__LINE__,"Unrecognized token after structure: %s\n",token);
}

void Structure::reset(char *line,char *token,System *system)
{
  delete system->structure;
  system->structure=NULL;
}

void Structure::file(char *line,char *token,System *system)
{
  FILE *fp;
  io_nexta(line,token);
  if (strcmp(token,"psf")==0) {
    io_nexta(line,token);
    fp=fpopen(token,"r");
    add_structure_psf_file(fp);
    fclose(fp);
  } else {
    fatal(__FILE__,__LINE__,"Unsupported structure file format: %s\n",token);
  }
}

void Structure::parse_shake(char *line,char *token,System *system)
{
  io_nexta(line,token);
  if (strcmp(token,"hbond")==0) {
    shakeHbond=true;
  } else if (strcmp(token,"none")==0) {
    shakeHbond=false;
  } else {
    fatal(__FILE__,__LINE__,"Unrecognized token %s for structure shake selection. Try hbond or none\n",token);
  }
}

void Structure::parse_noe(char *line,char *token,System *system)
{
  io_nexta(line,token);
  if (strcmp(token,"reset")==0) {
    noeList.clear();
  } else {
    std::string iselection=token;
    std::string jselection=io_nexts(line);
    int is,ns,i,j;
    if (system->selections->selectionMap.count(iselection)!=1) {
      fatal(__FILE__,__LINE__,"Unrecognized first selection name %s for noe restraints\n",token);
    }
    ns=0;
    for (is=0; is<system->selections->selectionMap[iselection].boolCount; is++) {
      if (system->selections->selectionMap[iselection].boolSelection[is]) {
        i=is;
        ns++;
      }
    }
    if (ns!=1) fatal(__FILE__,__LINE__,"Expected 1 atom in first selection, found %d\n",ns);
    if (system->selections->selectionMap.count(jselection)!=1) {
      fatal(__FILE__,__LINE__,"Unrecognized second selection name %s for noe restraints\n",token);
    }
    ns=0;
    for (is=0; is<system->selections->selectionMap[jselection].boolCount; is++) {
      if (system->selections->selectionMap[jselection].boolSelection[is]) {
        j=is;
        ns++;
      }
    }
    if (ns!=1) fatal(__FILE__,__LINE__,"Expected 1 atom in second selection, found %d\n",ns);
    struct NoePotential noe;
    noe.i=i;
    noe.j=j;
    noe.rmin=io_nextf(line)*ANGSTROM;
    noe.kmin=io_nextf(line)*KCAL_MOL/(ANGSTROM*ANGSTROM);
    noe.rmax=io_nextf(line)*ANGSTROM;
    noe.kmax=io_nextf(line)*KCAL_MOL/(ANGSTROM*ANGSTROM);
    noe.rpeak=io_nextf(line)*ANGSTROM;
    noe.rswitch=io_nextf(line)*ANGSTROM;
    noe.nswitch=io_nextf(line);
    noeList.push_back(noe);
  }
  noeCount=noeList.size();
}

void Structure::parse_harmonic(char *line,char *token,System *system)
{
  io_nexta(line,token);
  if (strcmp(token,"reset")==0) {
    harmList.clear();
  } else if (system->selections->selectionMap.count(token)==1) {
    std::string name=token;
    std::string massToken=io_nexts(line);
    bool massFlag;
    if (massToken=="mass") {
      massFlag=true;
    } else if (massToken=="none") {
      massFlag=false;
    } else {
      fatal(__FILE__,__LINE__,"Use mass or none for mass weighting scheme. Found unrecognized token %s\n",massToken.c_str());
    }
    real k=io_nextf(line);
    real n=io_nextf(line);
    int i;
    struct HarmonicPotential h;
    for (i=0; i<system->selections->selectionMap[name].boolCount; i++) {
      if (system->selections->selectionMap[name].boolSelection[i]) {
        h.idx=i;
        h.n=n;
        h.k=k*KCAL_MOL;
        h.k/=exp(n*log(ANGSTROM));
        if (massFlag) {
          h.k*=atomList[i].mass;
        }
        h.r0.x=system->coordinates->particlePosition[i][0];
        h.r0.y=system->coordinates->particlePosition[i][1];
        h.r0.z=system->coordinates->particlePosition[i][2];
        harmList.push_back(h);
      }
    }
  } else {
    fatal(__FILE__,__LINE__,"Unrecognized selection name %s for harmonic restraints\n",token);
  }
  harmCount=harmList.size();
}

void Structure::dump(char *line,char *token,System *system)
{
  fprintf(stdout,"%s:%d IMPLEMENT Structure::dump function.\n",__FILE__,__LINE__);
}

void Structure::add_structure_psf_file(FILE *fp)
{
  char line[MAXLENGTHSTRING];
  std::string token;
  int i,j,k;
  std::set<std::string> headerInfo;

  // "read" header"
  headerInfo.clear();
  fgets(line, MAXLENGTHSTRING, fp);
  token=io_nexts(line);
  if (strcmp(token.c_str(),"PSF")!=0) {
    fatal(__FILE__,__LINE__,"First line of PSF must start with PSF\n");
  }
  for (token=io_nexts(line); strcmp(token.c_str(),"")!=0; token=io_nexts(line)) {
    fprintf(stdout,"Reading PSF, found header string: '%s'\n",token.c_str());
    headerInfo.insert(token);
  }

  // "Read" title
  fgets(line, MAXLENGTHSTRING, fp);
  j=io_nexti(line,fp,"psf number of title lines");
  fprintf(stdout,"Reading PSF, expect !NTITLE: got %s",line);
  for (i=0; i<j; i++) {
    fgets(line, MAXLENGTHSTRING, fp);
  }

  // Read atoms
  fgets(line, MAXLENGTHSTRING, fp);
  atomCount=io_nexti(line,fp,"psf number of atoms");
  fprintf(stdout,"Reading PSF, expect !NATOM: got %s",line);
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
    at.resIdx=io_nexts(line);
    at.resName=io_nexts(line);
    at.atomName=io_nexts(line);
    at.atomTypeName=io_nexts(line);
    at.charge=io_nextf(line);
    at.mass=io_nextf(line);
    atomList.push_back(at);
  }

  // Read bonds
  fgets(line, MAXLENGTHSTRING, fp);
  bondCount=io_nexti(line,fp,"psf number of bonds");
  fprintf(stdout,"Reading PSF, expect !NBOND: bonds: got %s",line);
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
    bondList.push_back(bond);
  }
  
  // Read angles
  fgets(line, MAXLENGTHSTRING, fp);
  angleCount=io_nexti(line,fp,"psf number of angles");
  fprintf(stdout,"Reading PSF, expect !NTHETA: angles: got %s",line);
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
    angleList.push_back(angle);
  }
  
  // Read dihes
  fgets(line, MAXLENGTHSTRING, fp);
  diheCount=io_nexti(line,fp,"psf number of dihedrals");
  fprintf(stdout,"Reading PSF, expect !NPHI: dihedrals: got %s",line);
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
    diheList.push_back(dihe);
  }
  
  // Read imprs
  fgets(line, MAXLENGTHSTRING, fp);
  imprCount=io_nexti(line,fp,"psf number of impropers");
  fprintf(stdout,"Reading PSF, expect !NIMPHI: impropers: got %s",line);
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
    imprList.push_back(impr);
  }
  
  // Ignore donors
  fgets(line, MAXLENGTHSTRING, fp);
  j=io_nexti(line,fp,"psf number of donors");
  fprintf(stdout,"Reading PSF, expect !NDON: donors: got %s",line);
  for (i=0; i<2*j; i++) {
    io_nexti(line,fp,"psf donor atom");
  }
  
  // Ignore acceptors
  fgets(line, MAXLENGTHSTRING, fp);
  j=io_nexti(line,fp,"psf number of acceptors");
  fprintf(stdout,"Reading PSF, expect !NACC: acceptors: got %s",line);
  for (i=0; i<2*j; i++) {
    io_nexti(line,fp,"psf acceptor atom");
  }
  
  // Not even sure what this section is...
  fgets(line, MAXLENGTHSTRING, fp);
  j=io_nexti(line,fp,"psf nnb???");
  fprintf(stdout,"Reading PSF, expect !NNB: got %s",line);
  for (i=0; i<atomCount; i++) {
    io_nexti(line,fp,"psf nnb???");
  }

  // Or this one...
  fgets(line, MAXLENGTHSTRING, fp);
  j=io_nexti(line,fp,"psf ngrp???");
  fprintf(stdout,"Reading PSF, expect !NGRP NST2: got %s",line);
  for (i=0; i<3*j; i++) {
    io_nexti(line,fp,"psf ngrp???");
  }

  if (headerInfo.count("CHEQ")) {
    // OR this one...
    fgets(line, MAXLENGTHSTRING, fp);
    j=io_nexti(line,fp,"psf molnt???");
    fprintf(stdout,"Reading PSF, expect !MOLNT: got %s",line);
    for (i=0; i<atomCount; i++) {
      io_nexti(line,fp,"psf molnt???");
    }
  }

  // Read lone pairs
  fgets(line, MAXLENGTHSTRING, fp);
  i=io_nexti(line,fp,"psf lone pairs");
  j=io_nexti(line,fp,"psf lone pair hosts");
  fprintf(stdout,"Reading PSF, expect !NUMLP NUMLPH: got %s",line);
  if (i!=0 || j!=0) {
    int virtCount;
    int virtHostCount;
    bool w;
    real a,b,c;
    std::vector<int> virtHostList;
    // fatal(__FILE__,__LINE__,"Program is not set up to treat lone pairs. Found NUMLP=%d NUMLPH=%d in psf\n",i,j);
    virtCount=i;
    virtHostCount=j;
    virtHostList.clear();
    virt2List.clear();
    virt3List.clear();
    for (i=0; i<virtCount; i++) {
      j=io_nexti(line,fp,"psf lone pair host count");
      k=io_nexti(line,fp,"psf lone pair host pointer")-1;
      w=io_nextb(line); // "psf lone pair host weighting"
      a=io_nextf(line,fp,"psf lone pair value1");
      b=io_nextf(line,fp,"psf lone pair value2");
      c=io_nextf(line,fp,"psf lone pair value3");
      if (j==2 && w==0) { // Colinear lone pair
        struct VirtualSite2 virt2;
        virt2.vidx=k; // Stick the pointer to the host list here temporarily
        virt2.dist=a*ANGSTROM;
        virt2.scale=b;
        virt2List.push_back(virt2);
      } else if (j==3 && a!=0) { // (j==3 && a!=0 && (b==0 || b==180))
        // CHARMM lonepair.F90 currently has a bug for sin(b)!=0 2022-02-22
        struct VirtualSite3 virt3;
        virt3.vidx=k;
        virt3.dist=a*ANGSTROM;
        virt3.theta=b*DEGREES;
        virt3.phi=c*DEGREES;
        virt3List.push_back(virt3);
      } else {
        fatal(__FILE__,__LINE__,"Program found unsupported virtual site / lone pair type with %d hosts, and %d %d %f %f %f\n",j,k,(int)w,a,b,c);
      }
    }
    for (i=0; i<virtHostCount; i++) {
      j=io_nexti(line,fp,"psf lone pair host atom idx")-1;
      if (j>=atomCount || j<0) {
        fatal(__FILE__,__LINE__,"Atom %d in virtual site host %d is out of range\n",j,i);
      }
      virtHostList.push_back(j);
    }
    // Now collect all the atom indices
    virt2Count=virt2List.size();
    for (i=0; i<virt2Count; i++) {
      j=virt2List[i].vidx;
      virt2List[i].vidx=virtHostList[j];
      virt2List[i].hidx[0]=virtHostList[j+1];
      virt2List[i].hidx[1]=virtHostList[j+2];
    }
    virt3Count=virt3List.size();
    for (i=0; i<virt3Count; i++) {
      j=virt3List[i].vidx;
      virt3List[i].vidx=virtHostList[j];
      virt3List[i].hidx[0]=virtHostList[j+1];
      virt3List[i].hidx[1]=virtHostList[j+2];
      virt3List[i].hidx[2]=virtHostList[j+3];
    }
  }
  
  // Read cmaps
  if (headerInfo.count("CMAP")) {
    fgets(line, MAXLENGTHSTRING, fp);
    cmapCount=io_nexti(line,fp,"psf number of cmaps");
    fprintf(stdout,"Reading PSF, expect !NCRTERM: got %s",line);
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
      cmapList.push_back(cmap);
    }
  } else {
    cmapCount=0;
    cmapList.clear();
  }
}

void blade_init_structure(System *system)
{
  system+=omp_get_thread_num();
  if (system->structure) {
    delete(system->structure);
  }
  system->structure=new Structure();
}

void blade_dest_structure(System *system)
{
  system+=omp_get_thread_num();
  if (system->structure) {
    delete(system->structure);
  }
  system->structure=NULL;
}

void blade_add_atom(System *system,
  int atomIdx,const char *segName,const char *resIdx,const char *resName,
  const char *atomName,const char *atomTypeName,double charge,double mass)
{
  struct AtomStructure at;
  at.atomIdx=atomIdx-1;
  at.segName=segName;
  at.resIdx=resIdx;
  at.resName=resName;
  at.atomName=atomName;
  at.atomTypeName=atomTypeName;
  at.charge=charge;
  at.mass=mass;
  system+=omp_get_thread_num();
  system->structure->atomList.push_back(at);
  system->structure->atomCount=system->structure->atomList.size();
}

void blade_add_bond(System *system,int i,int j)
{
  struct Int2 bond;
  bond.i[0]=i-1;
  bond.i[1]=j-1;
  system+=omp_get_thread_num();
  system->structure->bondList.push_back(bond);
  system->structure->bondCount=system->structure->bondList.size();
}

void blade_add_angle(System *system,int i,int j,int k)
{
  struct Int3 angle;
  angle.i[0]=i-1;
  angle.i[1]=j-1;
  angle.i[2]=k-1;
  system+=omp_get_thread_num();
  system->structure->angleList.push_back(angle);
  system->structure->angleCount=system->structure->angleList.size();
}

void blade_add_dihe(System *system,int i,int j,int k,int l)
{
  struct Int4 dihe;
  dihe.i[0]=i-1;
  dihe.i[1]=j-1;
  dihe.i[2]=k-1;
  dihe.i[3]=l-1;
  system+=omp_get_thread_num();
  system->structure->diheList.push_back(dihe);
  system->structure->diheCount=system->structure->diheList.size();
}

void blade_add_impr(System *system,int i,int j,int k,int l)
{
  struct Int4 impr;
  impr.i[0]=i-1;
  impr.i[1]=j-1;
  impr.i[2]=k-1;
  impr.i[3]=l-1;
  system+=omp_get_thread_num();
  system->structure->imprList.push_back(impr);
  system->structure->imprCount=system->structure->imprList.size();
}

void blade_add_cmap(System *system,int i1,int j1,int k1,int l1,int i2,int j2,int k2,int l2)
{
  struct Int8 cmap;
  cmap.i[0]=i1-1;
  cmap.i[1]=j1-1;
  cmap.i[2]=k1-1;
  cmap.i[3]=l1-1;
  cmap.i[4]=i2-1;
  cmap.i[5]=j2-1;
  cmap.i[6]=k2-1;
  cmap.i[7]=l2-1;
  system+=omp_get_thread_num();
  system->structure->cmapList.push_back(cmap);
  system->structure->cmapCount=system->structure->cmapList.size();
}

void blade_add_virt2(System *system,int v,int h1,int h2,double dist,double scale)
{
  struct VirtualSite2 virt2;
  virt2.vidx=v-1;
  virt2.hidx[0]=h1-1;
  virt2.hidx[1]=h2-1;
  virt2.dist=dist*ANGSTROM;
  virt2.scale=scale;
  system+=omp_get_thread_num();
  system->structure->virt2List.push_back(virt2);
  system->structure->virt2Count=system->structure->virt2List.size();
}

void blade_add_virt3(System *system,int v,int h1,int h2,int h3,double dist,double theta,double phi)
{
  struct VirtualSite3 virt3;
  virt3.vidx=v-1;
  virt3.hidx[0]=h1-1;
  virt3.hidx[1]=h2-1;
  virt3.hidx[2]=h3-1;
  virt3.dist=dist*ANGSTROM;
  virt3.theta=theta*DEGREES;
  virt3.phi=phi*DEGREES;
  system+=omp_get_thread_num();
  system->structure->virt3List.push_back(virt3);
  system->structure->virt3Count=system->structure->virt3List.size();
}

void blade_add_shake(System *system,int shakeHbond)
{
  system+=omp_get_thread_num();
  system->structure->shakeHbond=shakeHbond;
}

void blade_add_noe(System *system,int i,int j,double rmin,double kmin,double rmax,double kmax,double rpeak,double rswitch,double nswitch)
{
  system+=omp_get_thread_num();
  struct NoePotential noe;
  noe.i=i-1;
  noe.j=j-1;
  noe.rmin=rmin*ANGSTROM;
  noe.kmin=kmin*KCAL_MOL/(ANGSTROM*ANGSTROM);
  noe.rmax=rmax*ANGSTROM;
  noe.kmax=kmax*KCAL_MOL/(ANGSTROM*ANGSTROM);
  noe.rpeak=rpeak*ANGSTROM;
  noe.rswitch=rswitch*ANGSTROM;
  noe.nswitch=nswitch;
  system->structure->noeList.push_back(noe);
  system->structure->noeCount=system->structure->noeList.size();
}

void blade_add_harmonic(System *system,int i,double k,double x0,double y0,double z0,double n)
{
  system+=omp_get_thread_num();
  struct HarmonicPotential h;
  h.idx=i-1;
  h.k=k;
  h.n=n;
  h.r0.x=x0;
  h.r0.y=y0;
  h.r0.z=z0;
  system->structure->harmList.push_back(h);
  system->structure->harmCount=system->structure->harmList.size();
}
