#ifndef SYSTEM_POTENTIAL_H
#define SYSTEM_POTENTIAL_H

// Forward declarations
class System;

class Potential {
  public:

  Potential() {
    fprintf(stdout,"IMPLEMENT Potential create %s %d\n",__FILE__,__LINE__);
  }

  ~Potential() {
  }

};

void parse_potential(char *line,System *system);

#endif
