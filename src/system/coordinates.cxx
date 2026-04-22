#include <omp.h>
#include <string.h>
#include <math.h>

#include "system/coordinates.h"
#include "main/blade_log.h"
#include "io/io.h"
#include "system/system.h"
#include "system/structure.h"
#include "rng/rng_cpu.h"
#include "system/state.h"



// Class constructors
Coordinates::Coordinates(int n,System *system) {
  atomCount=n;
  particleBoxABC.x=NAN;
  particlePosition=(real_x(*)[3])calloc(n,sizeof(real_x[3]));
  particleVelocity=(real_v(*)[3])calloc(n,sizeof(real_v[3]));

  setup_parse_coordinates();
}

Coordinates::~Coordinates() {
  if (particlePosition) free(particlePosition);
  if (particleVelocity) free(particleVelocity);
}



// Utility functions
bool operator<(const struct AtomCoordinates& a,const struct AtomCoordinates& b)
{
  return (a.segName<b.segName   || (a.segName==b.segName   && 
         (a.resIdx<b.resIdx     || (a.resIdx==b.resIdx     &&
         (a.atomName<b.atomName)))));
}

bool operator==(const struct AtomCoordinates& a,const struct AtomCoordinates& b)
{
  return a.segName==b.segName && a.resIdx==b.resIdx && a.atomName==b.atomName;
}



// Parsing functions
void parse_coordinates(char *line,System *system)
{
  char token[MAXLENGTHSTRING];
  std::string name;

  if (!system->coordinates) {
    if (system->structure && system->structure->atomCount>0) {
      system->coordinates=new Coordinates(system->structure->atomCount,system);
    } else {
      fatal(__FILE__,__LINE__,"Must finish creating atoms (use \"structure\" commands) before loading in positions with \"coordinates file\".\n");
    }
  }

  io_nexta(line,token);
  name=token;
  if (system->coordinates->parseCoordinates.count(name)==0) name="";
  // So much for function pointers being elegant.
  // call the function pointed to by: system->coordinates->parseCoordinates[name]
  // within the object: system->coordinates
  // with arguments: (line,token,system)
  (system->coordinates->*(system->coordinates->parseCoordinates[name]))(line,token,system);
}

void Coordinates::setup_parse_coordinates()
{
  parseCoordinates[""]=&Coordinates::error;
  helpCoordinates[""]="?> How did we get here?\n";
  parseCoordinates["reset"]=&Coordinates::reset;
  helpCoordinates["reset"]="?coordinates reset> This deletes the coordinates data structure.\n";
  parseCoordinates["file"]=&Coordinates::file;
  helpCoordinates["file"]="?coordinates file [pdb|crd] [filename]> This loads particle positions from the pdb or charmm crd filename\n";
  parseCoordinates["box"]=&Coordinates::parse_box;
  helpCoordinates["box"]="?coordinates box [name] [a b c alpha beta gamma]> This sets up the lattice vectors for the box. Acceptable names include cubi (cubic), tetr (tetragonal), orth (orthorhombic), mono (monoclinic), tric (triclinic), hexa (hexagonal), rhom (rhombohedral), octa (truncated octahedron), and rhdo (rhombic dodecahedron), see charmm crystal.doc for details. a is placed along the x axis, and b in the xy plane. alpha is the angle between b and c, beta is the angle between a and c, gamma is the angle between a and b. Units are angstroms and degrees.\n";
  parseCoordinates["velocity"]=&Coordinates::parse_velocity;
  helpCoordinates["velocity"]="?coordinates velocity [temperature]> This sets the velocities to a distribution centered on the specified temperature (in Kelvin)\n";
  parseCoordinates["print"]=&Coordinates::dump;
  helpCoordinates["print"]="?coordinates print> This prints selected contents of the coordinates data structure to standard out\n";
  parseCoordinates["help"]=&Coordinates::help;
  helpCoordinates["help"]="?coordinates help [directive]> Prints help on coordinates directive, including a list of subdirectives. If a subdirective is listed, this prints help on that specific subdirective.\n";
}

void Coordinates::help(char *line,char *token,System *system)
{
  char buf[256];
  std::string name=io_nexts(line);
  if (name=="") {
    blade_log("?coordinates > Available directives are:");
    for (std::map<std::string,std::string>::iterator ii=helpCoordinates.begin(); ii!=helpCoordinates.end(); ii++) {
      snprintf(buf, sizeof(buf), " %s", ii->first.c_str());
      blade_log(buf);
    }
    blade_log("\n");
  } else if (helpCoordinates.count(token)==1) {
    blade_log(helpCoordinates[name].c_str());
  } else {
    error(line,token,system);
  }
}

void Coordinates::error(char *line,char *token,System *system)
{
  fatal(__FILE__,__LINE__,"Unrecognized token after coordinates: %s\n",token);
}

void Coordinates::reset(char *line,char *token,System *system)
{
  delete system->coordinates;
  system->coordinates=NULL;
}

void Coordinates::file(char *line,char *token,System *system)
{
  FILE *fp;
  std::string fmt=io_nexts(line);

  io_nexta(line,token);
  fp=fpopen(token,"r");
  if (fmt=="pdb") {
    file_pdb(fp,system);
  } else if (fmt=="crd") {
    file_crd(fp,system);
  } else {
    fatal(__FILE__,__LINE__,"Unrecognized format %s. Options are pdb or crd.\n",fmt.c_str());
  }
  fclose(fp);
}

void Coordinates::dump(char *line,char *token,System *system)
{
  blade_log("coordinates print is not yet implemented (NYI)\n");
}

// ftp://ftp.wwpdb.org/pub/pdb/doc/format_descriptions/Format_v33_Letter.pdf
// 1-6 ATOM  /HETATM
// 7-11 atomIdx
// 13-16 atomName
// 17 ?
// 18-20 resName
// 22 chain
// 23-26 resIdx
// 27 resIdx insertion letter code
// 31-38 X 39-46 Y 47-54 Z
// 55-60 occupancy
// 61-66 temperature factor
// 77-78 element
// 79-80 charge
// CHARMM "PDB" format 73-76 segID
void Coordinates::file_pdb(FILE *fp,System *system)
{
  char line[MAXLENGTHSTRING];
  char token1[MAXLENGTHSTRING];
  char token2[MAXLENGTHSTRING];
  int i;
  struct AtomCoordinates as;
  double x; // Intentional double
  struct Real3 xyz;

  fileData.clear();
  while (fgets(line, MAXLENGTHSTRING, fp)!=NULL) {
    if (strncmp(line,"ATOM  ",6)==0 || strncmp(line,"HETATM",6)==0) {
      io_strncpy(token1,line+12,4);
      if (sscanf(token1,"%s",token2)!=1) fatal(__FILE__,__LINE__,"PDB error\n");
      as.atomName=token2;
      io_strncpy(token1,line+22,5);
      if (sscanf(token1,"%s",token2)!=1) fatal(__FILE__,__LINE__,"PDB error\n");
      as.resIdx=token2;
      io_strncpy(token1,line+72,4);
      if (sscanf(token1,"%s",token2)!=1) fatal(__FILE__,__LINE__,"PDB error\n");
      as.segName=token2;
      io_strncpy(token1,line+30,8);
      if (sscanf(token1,"%lf",&x)!=1) fatal(__FILE__,__LINE__,"PDBerror\n");
      xyz.i[0]=ANGSTROM*x;
      io_strncpy(token1,line+38,8);
      if (sscanf(token1,"%lf",&x)!=1) fatal(__FILE__,__LINE__,"PDBerror\n");
      xyz.i[1]=ANGSTROM*x;
      io_strncpy(token1,line+46,8);
      if (sscanf(token1,"%lf",&x)!=1) fatal(__FILE__,__LINE__,"PDBerror\n");
      xyz.i[2]=ANGSTROM*x;
      if (fileData.count(as)==0) {
        fileData[as]=xyz;
      }
    }
  }

  for (i=0; i<system->structure->atomList.size(); i++) { 
    as.segName=system->structure->atomList[i].segName;
    as.resIdx=system->structure->atomList[i].resIdx;
    as.atomName=system->structure->atomList[i].atomName;
    if (fileData.count(as)==1) {
      xyz=fileData[as];
      particlePosition[i][0]=xyz.i[0];
      particlePosition[i][1]=xyz.i[1];
      particlePosition[i][2]=xyz.i[2];
    } else {
      fatal(__FILE__,__LINE__,"Atom segName %s resIdx %s atomName %s missing from pdb file.\n",as.segName.c_str(),as.resIdx.c_str(),as.atomName.c_str());
    }
  }
}

// Charmm coor format
//         title
//         NATOM (I10)
//         ATOMNO RESNO   RES  TYPE  X     Y     Z   SEGID RESID Weighting
//           I10   I10 2X A8 2X A8       3F20.10     2X A8 2X A8 F20.10
void Coordinates::file_crd(FILE *fp,System *system)
{
  char line[MAXLENGTHSTRING];
  char token1[MAXLENGTHSTRING];
  char token2[MAXLENGTHSTRING];
  int i;
  struct AtomCoordinates as;
  double x; // Intentional double
  struct Real3 xyz;
  int extFmt=1; // no is 1, yes is 2
  char extStr[MAXLENGTHSTRING];
  extStr[0]=0;

  fileData.clear();
  while (fgets(line, MAXLENGTHSTRING, fp)!=NULL) {
    if (line[0]!='*') {
      sscanf(line,"%d %s",&i,extStr);
      if (i!=system->structure->atomCount) {
        fatal(__FILE__,__LINE__,"Wrong number of atoms in crd file %d, psf file contained %d atoms\nAtom count line says:\n%s\n",i,system->structure->atomCount,line);
      }
      if (strncmp(extStr,"EXT",3)==0) {
        extFmt=2;
      }
      break;
    }
  }
  while (fgets(line, MAXLENGTHSTRING, fp)!=NULL) {
    io_strncpy(token1,line+16*extFmt,4*extFmt); // 16-4 or 32-8
    if (sscanf(token1,"%s",token2)!=1) fatal(__FILE__,__LINE__,"CRD error\n");
    as.atomName=token2;
    io_strncpy(token1,line+56*extFmt,4*extFmt); // 56-4 or 112-8
    if (sscanf(token1,"%s",token2)!=1) fatal(__FILE__,__LINE__,"CRD error\n");
    as.resIdx=token2;
    io_strncpy(token1,line+51*extFmt,4*extFmt); // 51-4 or 102-8
    if (sscanf(token1,"%s",token2)!=1) fatal(__FILE__,__LINE__,"CRD error\n");
    as.segName=token2;

    io_strncpy(token1,line+20*extFmt,10*extFmt); // 20-10 or 40-20
    if (sscanf(token1,"%lf",&x)!=1) fatal(__FILE__,__LINE__,"CRDerror\n");
    xyz.i[0]=ANGSTROM*x;
    io_strncpy(token1,line+30*extFmt,10*extFmt); // 30-10 or 60-20
    if (sscanf(token1,"%lf",&x)!=1) fatal(__FILE__,__LINE__,"CRDerror\n");
    xyz.i[1]=ANGSTROM*x;
    io_strncpy(token1,line+40*extFmt,10*extFmt); // 40-10 or 80-20
    if (sscanf(token1,"%lf",&x)!=1) fatal(__FILE__,__LINE__,"CRDerror\n");
    xyz.i[2]=ANGSTROM*x;
    if (fileData.count(as)==0) {
      fileData[as]=xyz;
    }
  }

  for (i=0; i<system->structure->atomList.size(); i++) { 
    as.segName=system->structure->atomList[i].segName;
    as.resIdx=system->structure->atomList[i].resIdx;
    as.atomName=system->structure->atomList[i].atomName;
    if (fileData.count(as)==1) {
      xyz=fileData[as];
      particlePosition[i][0]=xyz.i[0];
      particlePosition[i][1]=xyz.i[1];
      particlePosition[i][2]=xyz.i[2];
    } else {
      fatal(__FILE__,__LINE__,"Atom segName %s resIdx %s atomName %s missing from crd file.\n",as.segName.c_str(),as.resIdx.c_str(),as.atomName.c_str());
    }
  }
}

void Coordinates::parse_box(char *line,char *token,System *system)
{
  int i,j;
  std::string nameString=io_nexts(line);
  if (nameString=="cubi") {
    particleBoxName=ebcubi;
  } else if (nameString=="tetr") {
    particleBoxName=ebtetr;
  } else if (nameString=="orth") {
    particleBoxName=eborth;
  } else if (nameString=="mono") {
    particleBoxName=ebmono;
  } else if (nameString=="tric") {
    particleBoxName=ebtric;
  } else if (nameString=="hexa") {
    particleBoxName=ebhexa;
  } else if (nameString=="rhom") {
    particleBoxName=ebrhom;
  } else if (nameString=="octa") {
    particleBoxName=ebocta;
  } else if (nameString=="rhdo") {
    particleBoxName=ebrhdo;
  } else {
    fatal(__FILE__,__LINE__,"Unrecognized box name: %s\n",nameString.c_str());
  }
  particleBoxABC.x=ANGSTROM*io_nextf(line);
  particleBoxABC.y=ANGSTROM*io_nextf(line);
  particleBoxABC.z=ANGSTROM*io_nextf(line);
  particleBoxAlBeGa.x=io_nextf(line);
  particleBoxAlBeGa.y=io_nextf(line);
  particleBoxAlBeGa.z=io_nextf(line);
}

void Coordinates::parse_velocity(char *line,char *token,System *system)
{
  int i,j;
  real T=io_nextf(line);
  for (i=0; i<atomCount; i++) {
    for (j=0; j<3; j++) {
      real m=system->structure->atomList[i].mass;
      particleVelocity[i][j]=sqrt(kB*T/m)*system->rngCPU->rand_normal();
      if (m==0) particleVelocity[i][j]=0;
    }
  }
}



void blade_init_coordinates(System *system,int n)
{
  system+=omp_get_thread_num();
  if (system->coordinates) {
    delete(system->coordinates);
  }
  system->coordinates=new Coordinates(n,system);
}

void blade_dest_coordinates(System *system)
{
  system+=omp_get_thread_num();
  if (system->coordinates) {
    delete(system->coordinates);
  }
  system->coordinates=NULL;
}

void blade_add_coordinates_position(System *system,int i,double x,double y,double z)
{
  system+=omp_get_thread_num();
  system->coordinates->particlePosition[i-1][0]=x;
  system->coordinates->particlePosition[i-1][1]=y;
  system->coordinates->particlePosition[i-1][2]=z;
}

void blade_add_coordinates_velocity(System *system,int i,double vx,double vy,double vz)
{
  system+=omp_get_thread_num();
  system->coordinates->particleVelocity[i-1][0]=vx;
  system->coordinates->particleVelocity[i-1][1]=vy;
  system->coordinates->particleVelocity[i-1][2]=vz;
}

void blade_add_coordinates_box(System *system,int name,double a,double b,double c,double alpha,double beta,double gamma)
{
  int i,j;
  system+=omp_get_thread_num();
  system->coordinates->particleBoxName=name;
  system->coordinates->particleBoxABC.x=a;
  system->coordinates->particleBoxABC.y=b;
  system->coordinates->particleBoxABC.z=c;
  system->coordinates->particleBoxAlBeGa.x=alpha;
  system->coordinates->particleBoxAlBeGa.y=beta;
  system->coordinates->particleBoxAlBeGa.z=gamma;
}
