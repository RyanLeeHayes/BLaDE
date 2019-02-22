#include <string.h>
#include <math.h>

#include "system/state.h"
#include "io/io.h"
#include "system/system.h"
#include "system/structure.h"
#include "rng/rng_cpu.h"
#include "rng/rng_gpu.h"



// Class constructors
State::State(int n,System *system) {
  int i,j;

  atomCount=n;
  box=(real(*)[3])calloc(3,sizeof(real[3]));
  box[0][0]=NAN;
  position=(real(*)[3])calloc(n,sizeof(real[3]));
#ifdef DOUBLE
  fposition=(float(*)[3])calloc(n,sizoef(float[3]));
#else
  fposition=position;
#endif
  velocity=(real(*)[3])calloc(n,sizeof(real[3]));
  force=(real(*)[3])calloc(n,sizeof(real[3]));
  mass=(real(*)[3])calloc(n,sizeof(real[3]));
  invsqrtMass=(real(*)[3])calloc(n,sizeof(real[3]));
  energy=(real*)calloc(eeend,sizeof(real));

  cudaMalloc(&(box_d),3*sizeof(real[3]));
  cudaMalloc(&(position_d),n*sizeof(real[3]));
  cudaMalloc(&(velocity_d),n*sizeof(real[3]));
  cudaMalloc(&(force_d),n*sizeof(real[3]));
  cudaMalloc(&(mass_d),n*sizeof(real[3]));
  cudaMalloc(&(invsqrtMass_d),n*sizeof(real[3]));
  cudaMalloc(&(random_d),2*n*sizeof(real[3]));
  cudaMalloc(&(energy_d),eeend*sizeof(real));

  for (i=0; i<n; i++) {
    for (j=0; j<3; j++) {
      mass[i][j]=system->structure->atomList[i].mass;
      invsqrtMass[i][j]=1.0/sqrt(mass[i][j]);
    }
  }

  rngCPU=new RngCPU;
  rngGPU=new RngGPU;

  setup_parse_state();
}

State::~State() {
  if (position) free(position);
#ifdef DOUBLE
  if (fposition) free(fposition);
#endif
  if (velocity) free(velocity);
  if (force) free(force);
  if (mass) free(mass);
  if (invsqrtMass) free(invsqrtMass);

  if (position_d) cudaFree(position_d);
  if (velocity_d) cudaFree(velocity_d);
  if (force_d) cudaFree(force_d);
  if (mass_d) cudaFree(mass_d);
  if (invsqrtMass_d) cudaFree(invsqrtMass_d);
  if (random_d) cudaFree(random_d);

  delete rngCPU;
  delete rngGPU;
}



// Utility functions
bool operator<(const struct AtomState& a,const struct AtomState& b)
{
  return (a.segName<b.segName   || (a.segName==b.segName   && 
         (a.resIdx<b.resIdx     || (a.resIdx==b.resIdx     &&
         (a.atomName<b.atomName)))));
}

bool operator==(const struct AtomState& a,const struct AtomState& b)
{
  return a.segName==b.segName && a.resIdx==b.resIdx && a.atomName==b.atomName;
}



// Parsing functions
void parse_state(char *line,System *system)
{
  char token[MAXLENGTHSTRING];
  std::string name;

  if (!system->state) {
    if (system->structure && system->structure->atomCount>0) {
      system->state=new State(system->structure->atomCount,system);
    } else {
      fatal(__FILE__,__LINE__,"Must finish creating atoms (use \"structure\" commands) before loading in positions with \"state file\".\n");
    }
  }

  io_nexta(line,token);
  name=token;
  if (system->state->parseState.count(name)==0) name="";
  // So much for function pointers being elegant.
  // call the function pointed to by: system->state->parseState[name]
  // within the object: system->state
  // with arguments: (line,token,system)
  (system->state->*(system->state->parseState[name]))(line,token,system);
}

void State::setup_parse_state()
{
  parseState[""]=&State::error;
  helpState[""]="?> How did we get here?\n";
  parseState["reset"]=&State::reset;
  helpState["reset"]="?state reset> This deletes the state data structure.\n";
  parseState["file"]=&State::file;
  helpState["file"]="?state file [filename]> This loads particle positions from the pdb filename\n";
  parseState["box"]=&State::parse_box;
  helpState["box"]="?state box [x1 y1 z1, x2 y2 z2, x3 y3 z3]> This loads the x y z cooridinates for the first, second, and third box vectors.\n";
  parseState["velocity"]=&State::parse_velocity;
  helpState["velocity"]="?state velocity [temperature]> This sets the velocities to a distribution centered on the specified temperature (in Kelvin)\n";
  parseState["print"]=&State::dump;
  helpState["print"]="?state print> This prints selected contents of the state data structure to standard out\n";
  parseState["help"]=&State::help;
  helpState["help"]="?state help [directive]> Prints help on state directive, including a list of subdirectives. If a subdirective is listed, this prints help on that specific subdirective.\n";
}

void State::help(char *line,char *token,System *system)
{
  std::string name=io_nexts(line);
  if (name=="") {
    fprintf(stdout,"?state > Available directives are:\n");
    for (std::map<std::string,std::string>::iterator ii=helpState.begin(); ii!=helpState.end(); ii++) {
      fprintf(stdout," %s",ii->first.c_str());
    }
    fprintf(stdout,"\n");
  } else if (helpState.count(token)==1) {
    fprintf(stdout,helpState[name].c_str());
  } else {
    error(line,token,system);
  }
}

void State::error(char *line,char *token,System *system)
{
  fatal(__FILE__,__LINE__,"Unrecognized token after state: %s\n",token);
}

void State::reset(char *line,char *token,System *system)
{
  delete system->state;
  system->state=NULL;
}

void State::file(char *line,char *token,System *system)
{
  FILE *fp;

  io_nexta(line,token);
  fp=fpopen(token,"r");
  file_pdb(fp,system);
  fclose(fp);
}

void State::dump(char *line,char *token,System *system)
{
  fprintf(stdout,"state print is not yet implemented (NYI)\n");
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
void State::file_pdb(FILE *fp,System *system)
{
  char line[MAXLENGTHSTRING];
  char token1[MAXLENGTHSTRING];
  char token2[MAXLENGTHSTRING];
  int i;
  struct AtomState as;
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

  // for (std::vector<struct AtomStructure>::iterator ii=system->structure->atomList.begin(); ii!=system->structure->atomList.end(); ii++)
  for (i=0; i<system->structure->atomList.size(); i++) { 
    as.segName=system->structure->atomList[i].segName;
    as.resIdx=system->structure->atomList[i].resIdx;
    as.atomName=system->structure->atomList[i].atomName;
    if (fileData.count(as)==1) {
      xyz=fileData[as];
      position[i][0]=xyz.i[0];
      position[i][1]=xyz.i[1];
      position[i][2]=xyz.i[2];
    }
  }
}

void State::parse_box(char *line,char *token,System *system)
{
  int i,j;
  for (i=0; i<3; i++) {
    for (j=0; j<3; j++) {
      box[i][j]=io_nextf(line);
      if (i!=j && box[i][j]!=0) {
        fatal(__FILE__,__LINE__,"Non-orthogonal boxes are not yet implemented NYI\n");
      }
    }
  }
  send_box();
  orthBox.x=box[0][0];
  orthBox.y=box[1][1];
  orthBox.z=box[2][2];
}

void State::parse_velocity(char *line,char *token,System *system)
{
  int i,j;
  real T=io_nextf(line);
  for (i=0; i<atomCount; i++) {
    for (j=0; j<3; j++) {
      velocity[i][j]=sqrt(kB*T/mass[i][j])*rngCPU->rand_normal();
    }
  }
}

void State::send_box() {
  cudaMemcpy(box_d,box,3*sizeof(real[3]),cudaMemcpyHostToDevice);
}
void State::recv_box() {
  cudaMemcpy(box,box_d,3*sizeof(real[3]),cudaMemcpyDeviceToHost);
}
void State::send_position() {
  cudaMemcpy(position_d,position,atomCount*sizeof(real[3]),cudaMemcpyHostToDevice);
}
void State::recv_position() {
  cudaMemcpy(position,position_d,atomCount*sizeof(real[3]),cudaMemcpyDeviceToHost);
}
void State::send_velocity() {
  cudaMemcpy(velocity_d,velocity,atomCount*sizeof(real[3]),cudaMemcpyHostToDevice);
}
void State::recv_velocity() {
  cudaMemcpy(velocity,velocity_d,atomCount*sizeof(real[3]),cudaMemcpyDeviceToHost);
}
void State::send_invsqrtMass() {
  cudaMemcpy(invsqrtMass_d,invsqrtMass,atomCount*sizeof(real[3]),cudaMemcpyHostToDevice);
}
void State::recv_invsqrtMass() {
  cudaMemcpy(invsqrtMass,invsqrtMass_d,atomCount*sizeof(real[3]),cudaMemcpyDeviceToHost);
}
void State::send_energy() {
  cudaMemcpy(energy_d,energy,eeend*sizeof(real),cudaMemcpyHostToDevice);
}
void State::recv_energy() {
  cudaMemcpy(energy,energy_d,eeend*sizeof(real),cudaMemcpyDeviceToHost);
}
