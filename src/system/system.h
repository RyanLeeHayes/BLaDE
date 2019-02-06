#ifndef SYSTEM_SYSTEM_H
#define SYSTEM_SYSTEM_H

#include "system/structure.h"
#include "system/parameters.h"
#include "system/state.h"
#include "msld/msld.h"

class System {
  public:
    Structure* structure;
    Parameters* parameters; // TEMPNOTE look in to std::map for atom types
    Msld* msld;
    State* state;

  System() {
    structure=NULL;
    parameters=NULL;
    msld=NULL;
    state=NULL;
  }

  ~System() {
    delete(structure);
    delete(parameters);
    delete(msld);
    delete(state);
  }

};

#endif
