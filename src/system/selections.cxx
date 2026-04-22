#include <string.h>
#include <stdio.h>

#include <string>
#include <map>
#include <set>

#include "main/defines.h"
#include "main/blade_log.h"
#include "io/io.h"
#include "system/system.h"
#include "io/variables.h"
#include "system/selections.h"
#include "system/structure.h"

void Selections::erase(const char *token)
{
  std::string name;
  name=token;
  if (selectionMap.count(name)==1) {
    selectionMap.erase(name);
  } else {
    fatal(__FILE__,__LINE__,"Did not find specified selection name %s\n",token);
  }
}

void Selections::insert(char *line,char *token,Structure *structure)
{
  std::string name;
  name=token;
  if (selectionMap.size()>=limit) {
    fatal(__FILE__,__LINE__,"Cannot define new selection because limit is exceeded. Use selection limit [int] to raise limit or selection delete [name] to delete selections\n");
  // } else if (selectionMap.count(name)==1) {
  //   fatal(__FILE__,__LINE__,"Cannot define selection with name %s, one already exists. Use selection delete [name] to remove it.\n",token);
  } else {
    char buf[256];
    snprintf(buf, sizeof(buf), "SELECTION> add %s to selections after parsing %s\n", name.c_str(), line);
    blade_log(buf);
    selectionMap[name]=parse_selection_string(line,structure);
  }
}

void Selections::count(char *line,char *token,System *system)
{
  Selection s1;
  std::string name,key;
  char value[MAXLENGTHSTRING];
  int i,N;

  name=token;
  if (selectionMap.count(name)==1) {
    s1=selectionMap[name];
    N=0;
    for (i=0; i<s1.boolCount; i++) {
      if (s1.boolSelection[i]) {
        N++;
      }
    }
  } else {
    fatal(__FILE__,__LINE__,"Selection %d does not exist\n",token);
  }

  key=io_nexts(line);
  sprintf(value,"%d",N);
  system->variables->data[key]=value;
}

bool search_match(std::string a,std::string b)
{
  int i;
  bool r=false;
  if (b=="*") {
    r=true;
  } else if (a.length()==b.length() && b.length()>0) {
    r=true;
    for (i=0; i<b.length(); i++) {
      if (!(b.c_str()[i]=='%' || b.c_str()[i]==a.c_str()[i])) {
        r=false;
      }
    }
  } else {
    r=false;
  }
  return r;
}

Selection Selections::parse_selection_string(char *line,Structure *structure)
{
  Selection s1;
  char token[MAXLENGTHSTRING];
  int i;
  int N=structure->atomList.size();

  std::set<std::string> knownTokens;
  knownTokens.insert("not");
  knownTokens.insert("and");
  knownTokens.insert("or");
  knownTokens.insert("none");
  knownTokens.insert("all");
  knownTokens.insert("hydrogen");
  knownTokens.insert("segid");
  knownTokens.insert("resid");
  knownTokens.insert("resids");
  knownTokens.insert("residrange");
  knownTokens.insert("resname");
  knownTokens.insert("atomname");
  knownTokens.insert("atomnames");
  knownTokens.insert("atom");
  knownTokens.insert("atomsearch");
  knownTokens.insert("selection");
  knownTokens.insert("");

  s1.sel_init(N);

  io_nexta(line,token);
  if (strcmp(token,"not")==0) {
    s1=parse_selection_string(line,structure);
    for (i=0; i<N; i++) {
      s1.boolSelection[i]=(!s1.boolSelection[i]);
    }
  } else if (strcmp(token,"and")==0) {
    s1=parse_selection_string(line,structure);
    Selection s2;
    s2=parse_selection_string(line,structure);
    for (i=0; i<N; i++) {
      s1.boolSelection[i]=(s1.boolSelection[i] && s2.boolSelection[i]);
    }
  } else if (strcmp(token,"or")==0) {
    s1=parse_selection_string(line,structure);
    Selection s2;
    s2=parse_selection_string(line,structure);
    for (i=0; i<N; i++) {
      s1.boolSelection[i]=(s1.boolSelection[i] || s2.boolSelection[i]);
    }
  } else if (strcmp(token,"none")==0) {
    for (i=0; i<N; i++) {
      s1.boolSelection[i]=false;
    }
  } else if (strcmp(token,"all")==0) {
    for (i=0; i<N; i++) {
      s1.boolSelection[i]=true;
    }
  } else if (strcmp(token,"hydrogen")==0) {
    for (i=0; i<N; i++) {
      s1.boolSelection[i]=(structure->atomList[i].atomTypeName[0]=='H');
    }
  } else if (strcmp(token,"segid")==0) {
    std::string segid=io_nexts(line);
    for (i=0; i<N; i++) {
      s1.boolSelection[i]=(structure->atomList[i].segName==segid);
    }
  } else if (strcmp(token,"resid")==0) {
    std::string resid=io_nexts(line);
    for (i=0; i<N; i++) {
      s1.boolSelection[i]=(structure->atomList[i].resIdx==resid);
    }
  } else if (strcmp(token,"resids")==0) {
    std::set<std::string> resids;
    resids.clear();
    while (knownTokens.count(io_peeks(line))==0) { // still a resid
      resids.insert(io_nexts(line));
    }
    for (i=0; i<N; i++) {
      s1.boolSelection[i]=(resids.count(structure->atomList[i].resIdx));
    }
  } else if (strcmp(token,"residrange")==0) {
    int resid1=io_nexti(line);
    int resid2=io_nexti(line);
    for (i=0; i<N; i++) {
      int resid;
      sscanf(structure->atomList[i].resIdx.c_str(),"%d",&resid);
      s1.boolSelection[i]=(resid>=resid1 && resid<=resid2);
    }
  } else if (strcmp(token,"resname")==0) {
    std::string resname=io_nexts(line);
    for (i=0; i<N; i++) {
      s1.boolSelection[i]=(structure->atomList[i].resName==resname);
    }
  } else if (strcmp(token,"atomname")==0) {
    std::string atomname=io_nexts(line);
    for (i=0; i<N; i++) {
      s1.boolSelection[i]=(structure->atomList[i].atomName==atomname);
    }
  } else if (strcmp(token,"atomnames")==0) {
    std::set<std::string> atomnames;
    atomnames.clear();
    while (knownTokens.count(io_peeks(line))==0) { // still an atom name
      atomnames.insert(io_nexts(line));
    }
    for (i=0; i<N; i++) {
      s1.boolSelection[i]=(atomnames.count(structure->atomList[i].atomName));
    }
  } else if (strcmp(token,"atom")==0) {
    std::string segid=io_nexts(line);
    std::string resid=io_nexts(line);
    std::string atomname=io_nexts(line);
    for (i=0; i<N; i++) {
      s1.boolSelection[i]=(structure->atomList[i].segName==segid &&
                           structure->atomList[i].resIdx==resid &&
                           structure->atomList[i].atomName==atomname);
    }
  } else if (strcmp(token,"atomsearch")==0) {
    std::string segid=io_nexts(line);
    std::string resid=io_nexts(line);
    std::string atomname=io_nexts(line);
    for (i=0; i<N; i++) {
      s1.boolSelection[i]=(search_match(structure->atomList[i].segName,segid) &&
                           search_match(structure->atomList[i].resIdx,resid) &&
                           search_match(structure->atomList[i].atomName,atomname));
    }
  } else if (strcmp(token,"selection")==0) {
    std::string selName=io_nexts(line);
    if (selectionMap.count(selName)==0) {
      fatal(__FILE__,__LINE__,"Selection token %s not found for copy\n",selName.c_str());
    }
    s1=selectionMap[selName];
    /*for (i=0; i<N; i++) {
      s1.boolSelection[i]=selectionMap[token].boolSelection[i];
    }*/
  } else {
    fatal(__FILE__,__LINE__,"Unrecognized token %s\n",token);
  }
  return s1;
}

void Selections::dump()
{
  int i,trueCount;
  char tag[]="PRINT SELECTIONS>";
  char buf[256];

  snprintf(buf, sizeof(buf), "%s selectionCount=%lu\n", tag, (unsigned long)selectionMap.size());
  blade_log(buf);
  snprintf(buf, sizeof(buf), "%s\n", tag);
  blade_log(buf);

  for (std::map<std::string,Selection>::iterator ii=selectionMap.begin(); ii!=selectionMap.end(); ii++) {
    trueCount=0;
    for (i=0; i<ii->second.boolCount; i++) {
      if (ii->second.boolSelection[i]) {
        trueCount++;
      }
    }
    snprintf(buf, sizeof(buf), "%s selection[%s]={contains %d atoms}\n", tag, ii->first.c_str(), trueCount);
    blade_log(buf);
  }
  snprintf(buf, sizeof(buf), "%s\n", tag);
  blade_log(buf);

}

void parse_selection(char *line,System *system)
{
  char token[MAXLENGTHSTRING];

  if (system->structure==NULL) {
    fatal(__FILE__,__LINE__,"selections cannot be defined until structure has been defined\n");
  }

  if (system->selections==NULL) {
    system->selections=new Selections;
  }

  io_nexta(line,token);
  if (strcmp(token,"reset")==0) {
    if (system->selections) {
      delete(system->selections);
      system->selections=NULL;
    }
  } else if (strcmp(token,"delete")==0) {
    io_nexta(line,token);
    system->selections->erase(token);
  } else if (strcmp(token,"define")==0) {
    io_nexta(line,token);
    system->selections->insert(line,token,system->structure);
  } else if (strcmp(token,"count")==0) {
    io_nexta(line,token);
    system->selections->count(line,token,system);
  } else if (strcmp(token,"limit")==0) {
    system->selections->limit=io_nexti(line);
    char buf[256];
    snprintf(buf, sizeof(buf), "New selection limit set to %d (be careful, selections can take up a lot of memory. Use selection delete [name] when done with a selection.)\n", system->selections->limit);
    blade_log(buf);
  } else if (strcmp(token,"print")==0) {
    system->selections->dump();
  } else {
    fatal(__FILE__,__LINE__,"Unrecognized selection token: %s\n",token);
  }
}
