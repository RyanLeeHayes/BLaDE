#include "system/system.h"
#include "io/io.h"

int main(int argc, char *argv[])
{
  System *system;
  FILE *fp;

  system=new(System);

  // open input file
  if (argc < 2) {
    fatal(__FILE__,__LINE__,"Need input file\n");
  }
  interpretter(argv[1],system,1);

  delete(system);
}
 // NYI - debug directive that allows the debugger to attach
