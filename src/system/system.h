#ifndef SYSTEM_SYSTEM_H
#define SYSTEM_SYSTEM_H

#include "system/parameters.h"
#include "system/structure.h"
#include "system/selections.h"
#include "msld/msld.h"
#include "system/potential.h"
#include "system/state.h"

// Forward declarations - not allowed when calling delete on objects

class System {
  public:
    Parameters* parameters; // TEMPNOTE look in to std::map for atom types
    Structure* structure;
    Selections* selections;
    Msld* msld;
    Potential* potential;
    State* state;

  System() {
    parameters=NULL;
    structure=NULL;
    selections=NULL;
    msld=NULL;
    potential=NULL;
    state=NULL;
  }

  ~System() {
    if (parameters) delete(parameters);
    if (structure) delete(structure);
    if (selections) delete(selections);
    if (msld) delete(msld);
    if (potential) delete(potential);
    if (state) delete(state);
  }
};

#endif
