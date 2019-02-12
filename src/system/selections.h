#ifndef SYSTEM_SELECTIONS_H
#define SYSTEM_SELECTIONS_H

#include "io/io.h"

// Forward declarations
class Structure;

class Selection {
  public:
#warning "Selections take up a lot of memory"
  // int length;
  // bool useBoolSelection;
  bool *boolSelection;
  int boolCount;
  // int *intSelection;
  // int intCount;

  Selection()
  {
    // length=0;
    boolCount=0;
    // useBoolSelection=0;
    boolSelection=NULL;
    // intSelection=NULL;
  }

  Selection(const Selection &other)
  {
    boolCount=other.boolCount;
    if (boolCount==0) {
      boolSelection==NULL;
    } else {
      boolSelection=new bool[boolCount];
      for (int i=0; i<boolCount; i++) {
        boolSelection[i]=other.boolSelection[i];
      }
    }
  }

  Selection& operator = (const Selection &other)
  {
    if (boolSelection) delete [] boolSelection;
    boolCount=other.boolCount;
    if (boolCount==0) {
      boolSelection==NULL;
    } else {
      boolSelection=new bool[boolCount];
      for (int i=0; i<boolCount; i++) {
        boolSelection[i]=other.boolSelection[i];
      }
    }
  }

  ~Selection()
  {
    if (boolSelection) delete [] boolSelection;
    boolCount=0;
    boolSelection=NULL;
    // if (intSelection) delete [] intSelection;
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
  void erase(const char *token);
  Selection parse_selection_string(char *line,Structure *structure);
  void dump();
};

void parse_selection(char *line,System *system);

#endif
