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

  void calc_force(int step,System *system);
};

void parse_potential(char *line,System *system);

#endif
