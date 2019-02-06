#ifndef SYSTEM_STRUCTURE_H
#define SYSTEM_STRUCTURE_H

#include <stdio.h>

class Structure {
  public:

  Structure() {
    fprintf(stdout,"IMPLEMENT Structure create %s %d\n",__FILE__,__LINE__);
  }

  ~Structure() {
  }

};

#endif
