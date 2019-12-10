#include <string.h>
#include <math.h>

#include "system/coordinates.h"
#include "io/io.h"
#include "system/system.h"
#include "system/structure.h"
#include "rng/rng_cpu.h"



// Class constructors
Coordinates::Coordinates(int n,System *system) {
  int i,j;

  atomCount=n;
  particleBox=(real(*)[3])calloc(3,sizeof(real[3]));
  particleBox[0][0]=NAN;
  particlePosition=(real(*)[3])calloc(n,sizeof(real[3]));
  particleVelocity=(real(*)[3])calloc(n,sizeof(real[3]));

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
  helpCoordinates["file"]="?coordinates file [filename]> This loads particle positions from the pdb filename\n";
  parseCoordinates["box"]=&Coordinates::parse_box;
  helpCoordinates["box"]="?coordinates box [x1 y1 z1, x2 y2 z2, x3 y3 z3]> This loads the x y z cooridinates for the first, second, and third box vectors. Input in Angstroms.\n";
  parseCoordinates["velocity"]=&Coordinates::parse_velocity;
  helpCoordinates["velocity"]="?coordinates velocity [temperature]> This sets the velocities to a distribution centered on the specified temperature (in Kelvin)\n";
  parseCoordinates["print"]=&Coordinates::dump;
  helpCoordinates["print"]="?coordinates print> This prints selected contents of the coordinates data structure to standard out\n";
  parseCoordinates["help"]=&Coordinates::help;
  helpCoordinates["help"]="?coordinates help [directive]> Prints help on coordinates directive, including a list of subdirectives. If a subdirective is listed, this prints help on that specific subdirective.\n";
}

void Coordinates::help(char *line,char *token,System *system)
{
  std::string name=io_nexts(line);
  if (name=="") {
    fprintf(stdout,"?coordinates > Available directives are:\n");
    for (std::map<std::string,std::string>::iterator ii=helpCoordinates.begin(); ii!=helpCoordinates.end(); ii++) {
      fprintf(stdout," %s",ii->first.c_str());
    }
    fprintf(stdout,"\n");
  } else if (helpCoordinates.count(token)==1) {
    fprintf(stdout,helpCoordinates[name].c_str());
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

  io_nexta(line,token);
  fp=fpopen(token,"r");
  file_pdb(fp,system);
  fclose(fp);
}

void Coordinates::dump(char *line,char *token,System *system)
{
  fprintf(stdout,"coordinates print is not yet implemented (NYI)\n");
}

// ftp://ftp.wwpdb.org/pub/pdb/doc/format_descriptions/Format_v33_Letter.pdf
// 1-6 ATOM  /HETATM
// 7-11 atomIdx
// 13-16 atomName
// 17 ?
// 18-20 resName
// 22 chain
// 23-26 resIdx
// 27?
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
      if (sscanf(line+21,"%d",&i)!=1) fatal(__FILE__,__LINE__,"PDB error\n");
      as.resIdx=i;
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
    }
  }
}

void Coordinates::parse_box(char *line,char *token,System *system)
{
  int i,j;
  for (i=0; i<3; i++) {
    for (j=0; j<3; j++) {
      particleBox[i][j]=ANGSTROM*io_nextf(line);
      if (i!=j && particleBox[i][j]!=0) {
        fatal(__FILE__,__LINE__,"Non-orthogonal boxes are not yet implemented NYI\n");
      }
    }
  }
  particleOrthBox.x=particleBox[0][0];
  particleOrthBox.y=particleBox[1][1];
  particleOrthBox.z=particleBox[2][2];
}

void Coordinates::parse_velocity(char *line,char *token,System *system)
{
  int i,j;
  real T=io_nextf(line);
  for (i=0; i<atomCount; i++) {
    for (j=0; j<3; j++) {
      real m=system->structure->atomList[i].mass;
      particleVelocity[i][j]=sqrt(kB*T/m)*system->rngCPU->rand_normal();
    }
  }
}



void blade_init_coordinates(System *system,int n)
{
  for (int id=0; id<system->idCount; id++) {
    if (system->coordinates) {
      delete(system->coordinates);
    }
    system->coordinates=new Coordinates(n,system);
    system++;
  }
}

void blade_dest_coordinates(System *system)
{
  for (int id=0; id<system->idCount; id++) {
    if (system->coordinates) {
      delete(system->coordinates);
    }
    system->coordinates=NULL;
    system++;
  }
}

void blade_add_coordinates_position(System *system,int i,double x,double y,double z)
{
  for (int id=0; id<system->idCount; id++) {
    system->coordinates->particlePosition[i][0]=x;
    system->coordinates->particlePosition[i][1]=y;
    system->coordinates->particlePosition[i][2]=z;
    system++;
  }
}

void blade_add_coordinates_velocity(System *system,int i,double vx,double vy,double vz)
{
  for (int id=0; id<system->idCount; id++) {
    system->coordinates->particleVelocity[i][0]=vx;
    system->coordinates->particleVelocity[i][1]=vy;
    system->coordinates->particleVelocity[i][2]=vz;
    system++;
  }
}

void blade_add_coordinates_box(System *system,double ax,double ay,double az,double bx,double by,double bz,double cx,double cy,double cz)
{
  int i,j;
  for (int id=0; id<system->idCount; id++) {
    system->coordinates->particleBox[0][0]=ax;
    system->coordinates->particleBox[0][1]=ay;
    system->coordinates->particleBox[0][2]=az;
    system->coordinates->particleBox[1][0]=bx;
    system->coordinates->particleBox[1][1]=by;
    system->coordinates->particleBox[1][2]=bz;
    system->coordinates->particleBox[2][0]=cx;
    system->coordinates->particleBox[2][1]=cy;
    system->coordinates->particleBox[2][2]=cz;
    for (i=0; i<3; i++) {
      for (j=0; j<3; j++) {
        if (i!=j && system->coordinates->particleBox[i][j]!=0) {
          fatal(__FILE__,__LINE__,"Non-orthogonal boxes are not yet implemented NYI\n");
        }
      }
    }
    system->coordinates->particleOrthBox.x=system->coordinates->particleBox[0][0];
    system->coordinates->particleOrthBox.y=system->coordinates->particleBox[1][1];
    system->coordinates->particleOrthBox.z=system->coordinates->particleBox[2][2];
    system++;
  }
}
