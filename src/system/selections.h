#ifndef SYSTEM_SELECTIONS_H
#define SYSTEM_SELECTIONS_H

#include <string>
#include <map>

// Forward declarations
class System;
class Structure;

class Selection {
  public:
  bool *boolSelection;
  int boolCount;

  Selection()
  {
    boolCount=0;
    boolSelection=NULL;
  }

  Selection(const Selection &other)
  {
    boolCount=other.boolCount;
    if (boolCount==0) {
      boolSelection=NULL;
    } else {
      boolSelection=new bool[boolCount];
      for (int i=0; i<boolCount; i++) {
        boolSelection[i]=other.boolSelection[i];
      }
    }
  }

  Selection operator=(const Selection &other)
  {
    if (boolSelection) delete [] boolSelection;
    boolCount=other.boolCount;
    if (boolCount==0) {
      boolSelection=NULL;
    } else {
      boolSelection=new bool[boolCount];
      for (int i=0; i<boolCount; i++) {
        boolSelection[i]=other.boolSelection[i];
      }
    }
    return *this;
  }

  ~Selection()
  {
    if (boolSelection) delete [] boolSelection;
    boolCount=0;
    boolSelection=NULL;
  }

  void sel_init(int l)
  {
    boolCount=l;
    boolSelection=new bool[l];
  }
};

class Selections {
  public:
  std::map<std::string,Selection> selectionMap;
  int limit;

  Selections ()
  {
    limit=100;
  }

  ~Selections ()
  {
    ;
  }

  void insert(char *line,char *token,Structure *structure);
  void count(char *line,char *token,System *system);
  void erase(const char *token);
  Selection parse_selection_string(char *line,Structure *structure);
  void dump();
};

void parse_selection(char *line,System *system);

#endif
