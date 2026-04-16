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
  boRestCount=0;
  boRestList.clear();
  anRestCount=0;
  anRestList.clear();
  diRestCount=0;
  diRestList.clear();
  resdCount=0;      // eeresd
  resdList.clear(); // eeresd
  MLPModelCount=0;  // eemlp
  MLPList.clear();  // eemlp

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
  parseStructure["dihedral"]=&Structure::parse_diRest;
  helpStructure["dihedral"]="[selection] [selection] [selection] [selection] [kconst] [angle0] [periodicity]\n";
  parseStructure["resd"]=&Structure::parse_resd; //eeresd
  helpStructure["resd"]="?structure resd [selection_dist1_atom1] [selection_dist1_atom2] [selection_dist2_atom1] [selection_dist2_atom2] [dist1_coefficient] [dist2_coefficient] [ref_dist_diff] [k_dist_diff]> Apply a CHARMM-style RESD restraint with linear combination of two distances (dist1 and dist2)\n"; // eeresd
  parseStructure["mlp"]=&Structure::parse_mlp; //eemlp
  helpStructure["mlp"]="?structure mlp [atom selection] [tani] [TorchScript .pt filename] > Apply Torch-derived forces to selected atoms based on the given model TYPE (eg. tani) and TorchScript FILE \n"; // eemlp
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

void Structure::parse_diRest(char *line,char *token,System *system)
{
  io_nexta(line,token);
  if (strcmp(token,"reset")==0) {
    diRestList.clear();
  } else {
    std::string iselection=token;
    std::string jselection=io_nexts(line);
    std::string kselection=io_nexts(line);
    std::string lselection=io_nexts(line); 
    int is,ns,i,j,k,l;
    if (system->selections->selectionMap.count(iselection)!=1) {
      fatal(__FILE__,__LINE__,"Unrecognized first selection name %s for dihedral restraints\n",token);
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
      fatal(__FILE__,__LINE__,"Unrecognized second selection name %s for dihedral restraints\n",token);
    }
    ns=0;
    for (is=0; is<system->selections->selectionMap[jselection].boolCount; is++) {
      if (system->selections->selectionMap[jselection].boolSelection[is]) {
        j=is;
        ns++;
      }
    }
    if (ns!=1) fatal(__FILE__,__LINE__,"Expected 1 atom in second selection, found %d\n",ns);
    if (system->selections->selectionMap.count(kselection)!=1){
      fatal(__FILE__,__LINE__,"Unrecognized third selection name %s for dihedral restraints\n",token);
    }
    ns=0;
    for (is=0; is<system->selections->selectionMap[kselection].boolCount; is++) {
       if (system->selections->selectionMap[kselection].boolSelection[is]) {
        k=is;
        ns++;
      }
    }
    if (ns!=1) fatal(__FILE__,__LINE__,"Expected 1 atom in third selection, found %d\n",ns);
    if (system->selections->selectionMap.count(lselection)!=1){
      fatal(__FILE__,__LINE__,"Unrecognized fourth selection name %s for dihedral restraints\n",token);
    }
    ns=0;
    for (is=0; is<system->selections->selectionMap[lselection].boolCount; is++) {
       if (system->selections->selectionMap[lselection].boolSelection[is]) {
        l=is;
        ns++;
      }
    }
    if (ns!=1) fatal(__FILE__,__LINE__,"Expected 1 atom in fourth selection, found %d\n",ns);
    struct DiRestPotential dr;
    dr.idx[0]=i;
    dr.idx[1]=j;
    dr.idx[2]=k;
    dr.idx[3]=l;
    dr.kphi=io_nextf(line)*KCAL_MOL;
    dr.phi0=io_nextf(line)*DEGREES;
    dr.nphi=io_nexti(line);
    dr.block=0; // Assume dihedral restraint is not scaled by lambda
    diRestList.push_back(dr);
    }
  diRestCount=diRestList.size();
}

// eeresd-begin
void Structure::parse_resd(char *line,char *token,System *system)
{
  io_nexta(line,token);
  if (strcmp(token,"reset")==0) {
    resdList.clear();
  } else {
    std::string i1selection=token;
    std::string i2selection=io_nexts(line);
    std::string j1selection=io_nexts(line);
    std::string j2selection=io_nexts(line);


    int is,ns,i1,i2,j1,j2;
    // sel 1 //distance1 atom1
    if (system->selections->selectionMap.count(i1selection)!=1) {
      fatal(__FILE__,__LINE__,"Unrecognized atom1_for_distance1 selection name %s for resd restraints\n",token);
    }
    ns=0;
    for (is=0; is<system->selections->selectionMap[i1selection].boolCount; is++) {
      if (system->selections->selectionMap[i1selection].boolSelection[is]) {
        i1=is;
        ns++;
      }
    }
    if (ns!=1) fatal(__FILE__,__LINE__,"Expected 1 atom in atom1_for_distance1 selection, found %d\n",ns);
    // sel 2 //distance1 atom2
    if (system->selections->selectionMap.count(i2selection)!=1) {
      fatal(__FILE__,__LINE__,"Unrecognized atom2_for_distance1 selection name %s for resd restraints\n",token);
    }
    ns=0;
    for (is=0; is<system->selections->selectionMap[i2selection].boolCount; is++) {
      if (system->selections->selectionMap[i2selection].boolSelection[is]) {
        i2=is;
        ns++;
      }
    }
    if (ns!=1) fatal(__FILE__,__LINE__,"Expected 1 atom in atom2_for_distance1 selection, found %d\n",ns);
    // sel 3 //distance2 atom1
        if (system->selections->selectionMap.count(j1selection)!=1) {
      fatal(__FILE__,__LINE__,"Unrecognized atom1_for_distance2 selection name %s for resd restraints\n",token);
    }
    ns=0;
    for (is=0; is<system->selections->selectionMap[j1selection].boolCount; is++) {
      if (system->selections->selectionMap[j1selection].boolSelection[is]) {
        j1=is;
        ns++;
      }
    }
    if (ns!=1) fatal(__FILE__,__LINE__,"Expected 1 atom in atom1_for_distance2 selection, found %d\n",ns);
    // sel 4 //distance2 atom2
        if (system->selections->selectionMap.count(j2selection)!=1) {
      fatal(__FILE__,__LINE__,"Unrecognized atom2_for_distance2 selection name %s for resd restraints\n",token);
    }
    ns=0;
    for (is=0; is<system->selections->selectionMap[j2selection].boolCount; is++) {
      if (system->selections->selectionMap[j2selection].boolSelection[is]) {
        j2=is;
        ns++;
      }
    }
    if (ns!=1) fatal(__FILE__,__LINE__,"Expected 1 atom in atom2_for_distance2 selection, found %d\n",ns);
    //
    struct ResdPotential resd;
    resd.i1=i1;
    resd.i2=i2;
    resd.j1=j1;
    resd.j2=j2;
    resd.ci=io_nextf(line);
    resd.cj=io_nextf(line);
    resd.rdist=io_nextf(line)*ANGSTROM;
    resd.kdist=io_nextf(line)*KCAL_MOL/(ANGSTROM*ANGSTROM);
    resdList.push_back(resd);
  }
  resdCount=resdList.size();
} // eeresd-end

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

// eemlp-begin
void Structure::parse_mlp(char *line,char *token,System *system)
{
  io_nexta(line,token);

  fprintf(stderr,"[parse_mlp] enter token=%s\n", token ? token : "(null)");

  if (strcmp(token,"reset")==0) {
    fprintf(stderr,"[parse_mlp] reset MLPList\n");
    MLPList.clear();

  } else if (system->selections->selectionMap.count(token)==1) {
    std::string name=token;
    std::string MLPTypeToken=io_nexts(line);
    std::string MLPFileToken;

    fprintf(stderr,"[parse_mlp] selection name=%s\n", name.c_str());
    fprintf(stderr,"[parse_mlp] boolCount=%d atomCount=%d\n",
            system->selections->selectionMap[name].boolCount,
            system->structure->atomCount);

    if (MLPTypeToken=="tani") {
      MLPFileToken=io_nexts(line);
      fprintf(stderr,"[parse_mlp] MLPTypeToken=%s MLPFileToken=%s\n",
              MLPTypeToken.c_str(), MLPFileToken.c_str());
    } else {
      fprintf(stderr,"[parse_mlp] bad MLPTypeToken=%s\n", MLPTypeToken.c_str());
      fatal(__FILE__,__LINE__,"Unrecognized MLP model type token %s. Currently only tani forces implemented.\n",MLPTypeToken.c_str());
    }

    struct MLPotential mlp;
    int i,nsel;
    std::vector<real> mlMassidx;

    mlp.is_tani=0;
    if (MLPTypeToken=="tani") { mlp.is_tani=1; }

    mlp.ptname    = MLPFileToken;
    mlp.mlnatoms  = system->structure->atomCount;

    fprintf(stderr,"[parse_mlp] before resize mlnatoms=%d boolCount=%d\n",
            mlp.mlnatoms,
            system->selections->selectionMap[name].boolCount);

    mlp.mlatomidx.resize(system->selections->selectionMap[name].boolCount);
    mlMassidx.resize(system->selections->selectionMap[name].boolCount);
    mlp.mlZidx.resize(system->selections->selectionMap[name].boolCount);
    mlp.mlmaskid.resize(system->selections->selectionMap[name].boolCount);

    fprintf(stderr,"[parse_mlp] after resize mlatomidx=%zu mlMassidx=%zu mlZidx=%zu mlmaskid=%zu\n",
            mlp.mlatomidx.size(), mlMassidx.size(), mlp.mlZidx.size(), mlp.mlmaskid.size());

    nsel=0;
    for (i=0; i<system->selections->selectionMap[name].boolCount; i++) {

      if (i < 5 || i == system->selections->selectionMap[name].boolCount-1) {
        fprintf(stderr,"[parse_mlp] scan i=%d selected=%d\n",
                i,
                (int)system->selections->selectionMap[name].boolSelection[i]);
      }

      if (system->selections->selectionMap[name].boolSelection[i]) {
        fprintf(stderr,"[parse_mlp] selected i=%d nsel=%d\n", i, nsel);

        mlp.mlmaskid[i]=1;
        mlp.mlatomidx[nsel]=i;

        fprintf(stderr,"[parse_mlp] reading atomList[%d]\n", i);
        mlMassidx[nsel]=system->structure->atomList[i].mass;

        fprintf(stderr,"[parse_mlp] atom %d mass=%f atomTypeName[0]=%c atomTypeName[1]=%c\n",
                i,
                (double)mlMassidx[nsel],
                system->structure->atomList[i].atomTypeName[0],
                system->structure->atomList[i].atomTypeName[1]);

        if (mlp.is_tani==1) {
                 if ((mlMassidx[nsel]==1.00800) || 
                     (system->structure->atomList[i].atomTypeName[0]=='H')) {
            mlp.mlZidx[nsel]=1;
          } else if ((mlMassidx[nsel]==12.01100) ||
                     (system->structure->atomList[i].atomTypeName[0]=='C')) {
            mlp.mlZidx[nsel]=6;
          } else if ((mlMassidx[nsel]==14.00700) ||
                     (system->structure->atomList[i].atomTypeName[0]=='N')) {
            mlp.mlZidx[nsel]=7;
          } else if ((mlMassidx[nsel]==15.99940) ||
                     (system->structure->atomList[i].atomTypeName[0]=='O')) {
            mlp.mlZidx[nsel]=8;
          } else if ((mlMassidx[nsel]==18.99800) ||
                     (system->structure->atomList[i].atomTypeName[0]=='F')) {
            mlp.mlZidx[nsel]=9;
          } else if ((mlMassidx[nsel]==32.06000) ||
                     (system->structure->atomList[i].atomTypeName[0]=='S')) {
            mlp.mlZidx[nsel]=16;
          } else if ((mlMassidx[nsel]==35.45300) ||
                     (system->structure->atomList[i].atomTypeName[0]=='C' &&
                      system->structure->atomList[i].atomTypeName[1]=='L')) {
            mlp.mlZidx[nsel]=17;
          } else {
            fprintf(stderr,"[parse_mlp] unsupported element i=%d nsel=%d mass=%f type0=%c type1=%c\n",
                    i, nsel, (double)mlMassidx[nsel],
                    system->structure->atomList[i].atomTypeName[0],
                    system->structure->atomList[i].atomTypeName[1]);
            fatal(__FILE__,__LINE__,"Unsupported element with mass %f for tani MLP. Only H, C, N, O, F, S, Cl supported.\n",mlMassidx[nsel]);
          }
        }

        fprintf(stderr,"[parse_mlp] assigned Z=%d for i=%d nsel=%d\n",
                mlp.mlZidx[nsel], i, nsel);

        nsel++;
      } else {
        mlp.mlmaskid[i]=0;
      }
    }

    fprintf(stderr,"[parse_mlp] loop done nsel=%d\n", nsel);

    mlp.ptnml=nsel;

    fprintf(stderr,"[parse_mlp] before compact resize ptnml=%d current mlatomidx=%zu mlMassidx=%zu mlZidx=%zu\n",
            mlp.ptnml,
            mlp.mlatomidx.size(), mlMassidx.size(), mlp.mlZidx.size());

    mlp.mlatomidx.resize(mlp.ptnml);
    mlMassidx.resize(mlp.ptnml);
    mlp.mlZidx.resize(mlp.ptnml);

    fprintf(stderr,"[parse_mlp] after compact resize mlatomidx=%zu mlMassidx=%zu mlZidx=%zu\n",
            mlp.mlatomidx.size(), mlMassidx.size(), mlp.mlZidx.size());

    fprintf(stderr,"[parse_mlp] before push_back MLPList.size=%zu\n", MLPList.size());
    MLPList.push_back(mlp);
    fprintf(stderr,"[parse_mlp] after push_back MLPList.size=%zu\n", MLPList.size());

  } else {
    fprintf(stderr,"[parse_mlp] unknown selection token=%s\n", token ? token : "(null)");
    fatal(__FILE__,__LINE__,"Unrecognized selection name %s for mlp\n",token);
  }

  MLPModelCount=MLPList.size();
  fprintf(stderr,"[parse_mlp] exit MLPModelCount=%d\n", MLPModelCount);
} // eemlp-end

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

void blade_add_borest(System *system,int i,int j,double kr,double r0,int lambdaBlock)
{
  system+=omp_get_thread_num();
  struct BoRestPotential br;
  br.idx[0]=i-1;
  br.idx[1]=j-1;
  br.kr=kr*(KCAL_MOL/ANGSTROM/ANGSTROM);
  br.r0=r0*ANGSTROM;
  br.block=lambdaBlock-1;
  system->structure->boRestList.push_back(br);
  system->structure->boRestCount=system->structure->boRestList.size();
}

void blade_add_anrest(System *system,int i,int j,int k,double kt,double t0,int lambdaBlock)
{
  system+=omp_get_thread_num();
  struct AnRestPotential ar;
  ar.idx[0]=i-1;
  ar.idx[1]=j-1;
  ar.idx[2]=k-1;
  ar.kt=kt*KCAL_MOL;
  ar.t0=t0*DEGREES;
  ar.block=lambdaBlock-1;
  system->structure->anRestList.push_back(ar);
  system->structure->anRestCount=system->structure->anRestList.size();
}

void blade_add_direst(System *system,int i,int j,int k,int l,double kphi,int nphi,double phi0,int lambdaBlock)
{
  system+=omp_get_thread_num();
  struct DiRestPotential dr;
  dr.idx[0]=i-1;
  dr.idx[1]=j-1;
  dr.idx[2]=k-1;
  dr.idx[3]=l-1;
  dr.kphi=kphi*KCAL_MOL;
  dr.phi0=phi0*DEGREES;
  dr.nphi = nphi;
  dr.block=lambdaBlock-1;
  system->structure->diRestList.push_back(dr);
  system->structure->diRestCount=system->structure->diRestList.size();
}

// eeresd-begin
void blade_add_resd(System *system, int i1, int i2, int j1, int j2, double ci, double cj, double rdist, double kdist)
{
  system+=omp_get_thread_num();
  struct ResdPotential resd;
  resd.i1=i1-1;
  resd.i2=i2-1;
  resd.j1=j1-1;
  resd.j2=j2-1;
  resd.ci=ci;
  resd.cj=cj;
  resd.rdist=rdist*ANGSTROM;
  resd.kdist=kdist*KCAL_MOL/(ANGSTROM*ANGSTROM);
  system->structure->resdList.push_back(resd);
  system->structure->resdCount=system->structure->resdList.size();
}  // eeresd-end

// eemlp-begin
void blade_tani_internal_setup(System *system, const int is_tani, const char *ptname, const int ptnml, 
                      const int *mlatomidx, const int *mlZidx,const int *mlmaskid, const int mlnatoms)
{
  system += omp_get_thread_num();
  struct MLPotential mlp;
  mlp.ptname    = ptname ? std::string(ptname) : std::string();
  mlp.is_tani   = is_tani;    // 1 for TorchANI, -1 NOT TorchANI
  mlp.ptnml     = ptnml;      // number of QM/ML atoms (mlnm2_c)
  mlp.mlnatoms  = mlnatoms;   // natom_mlmm_c
  mlp.mlatomidx.resize(ptnml);
  mlp.mlZidx.resize(ptnml);
  for (int i = 0; i < ptnml; ++i) {
    // Fortran indices are 1-based → convert to 0-based for C++
    mlp.mlatomidx[i]  = mlatomidx[i] - 1;
    mlp.mlZidx[i] = mlZidx[i];
  }
  mlp.mlmaskid.resize(mlnatoms);
  for (int i = 0; i < mlnatoms; ++i) {
    mlp.mlmaskid[i] = mlmaskid[i];
  }
  system->structure->MLPList.clear();
  system->structure->MLPList.push_back(mlp);
  system->structure->MLPModelCount=system->structure->MLPList.size();
} // eemlp-end