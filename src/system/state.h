#ifndef SYSTEM_STATE_H
#define SYSTEM_STATE_H

#include <stdio.h>

class State {
  public:

  State() {
    fprintf(stdout,"IMPLEMENT State create %s %d\n",__FILE__,__LINE__);
  }

  ~State() {
  }

};

#endif
