#ifndef MSLD_MSLD_H
#define MSLD_MSLD_H

#include <stdio.h>

class Msld {
  public:

  Msld() {
    fprintf(stdout,"IMPLEMENT MSLD create %s %d\n",__FILE__,__LINE__);
  }

  ~Msld() {
  }

};

#endif
