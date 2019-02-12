#ifndef MSLD_MSLD_H
#define MSLD_MSLD_H

#include <stdio.h>

// Forward declarations
class System;

class Msld {
  public:

  Msld() {
    fprintf(stdout,"IMPLEMENT MSLD create %s %d\n",__FILE__,__LINE__);
  }

  ~Msld() {
  }

};

void parse_msld(char *line,System *system);

#endif
