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
    if (structure!=NULL) {
      delete(structure);
    }
    if (parameters!=NULL) {
      delete(parameters);
    }
    if (msld!=NULL) {
      delete(msld);
    }
    if (state!=NULL) {
      delete(state);
    }
  }

};

#endif
