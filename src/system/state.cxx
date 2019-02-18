#include <string.h>

#include "system/system.h"
#include "io/io.h"
#include "system/state.h"



// Utility functions
bool operator<(const struct AtomState& a,const struct AtomState& b)
{
  return (a.segName<b.segName   || (a.segName==b.segName   && 
         (a.resIdx<b.resIdx     || (a.resIdx==b.resIdx     &&
         (a.atomName<b.atomName)))));
}



// Parsing functions
void parse_state(char *line,System *system)
{
  char token[MAXLENGTHSTRING];
  std::string name;

  if (!system->state) {
    if (system->structure && system->structure->atomCount>0) {
      system->state=new State(system->structure->atomCount);
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
  while (fgets(line, MAXLENGTHSTRING, fp)==NULL) {
    if (strncmp(line,"ATOM  ",6)==0 || strncmp(line,"HETATM",6)==0) {
      strncpy(token1,line+12,4);
      if (sscanf(token1,"%s",token2)!=1) fatal(__FILE__,__LINE__,"PDB error\n");
      as.atomName=token2;
      if (sscanf(line+21,"%d",&i)!=1) fatal(__FILE__,__LINE__,"PDB error\n");
      as.resIdx=i;
      strncpy(token1,line+72,4);
      if (sscanf(token1,"%s",token2)!=1) fatal(__FILE__,__LINE__,"PDB error\n");
      as.segName=token2;
      if (sscanf(line+30,"%8.3lf",&x)!=1) fatal(__FILE__,__LINE__,"PDBerror\n");
      xyz.i[0]=x;
      if (sscanf(line+38,"%8.3lf",&x)!=1) fatal(__FILE__,__LINE__,"PDBerror\n");
      xyz.i[1]=x;
      if (sscanf(line+46,"%8.3lf",&x)!=1) fatal(__FILE__,__LINE__,"PDBerror\n");
      xyz.i[2]=x;
      if (fileData.count(as)==0) {
        fileData[as]=xyz;
      }
    }
  }

  // for (std::vector<struct AtomStructure>::iterator ii=system->structure->atomList.begin(); ii!=system->structure->atomList.end(); ii++)
  for (std::vector<struct AtomStructure>::iterator ii=system->structure->atomList.begin(); ii!=system->structure->atomList.end(); ii++) {
    as.segName=ii->segName;
    as.resIdx=ii->resIdx;
    as.atomName=ii->atomName;
    if (fileData.count(as)==1) {
      xyz=fileData[as];
      position[i][0]=xyz.i[0];
      position[i][1]=xyz.i[1];
      position[i][2]=xyz.i[2];
    }
    i++;
  }
}

void State::parse_box(char *line,char *token,System *system)
{
  int i,j;
  for (i=0; i<3; i++) {
    for (j=0; j<3; j++) {
      box[i][j]=io_nextf(line);
    }
  }
}
