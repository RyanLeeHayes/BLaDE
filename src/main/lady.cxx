#include <mpi.h>

#include "system/system.h"
#include "io/io.h"

int main(int argc, char *argv[])
{
  System *system;
  FILE *fp;

  MPI_Init(&argc,&argv);

  system=new(System);
  MPI_Comm_rank(MPI_COMM_WORLD,&system->id);
  MPI_Comm_size(MPI_COMM_WORLD,&system->idCount);

  // open input file
  if (argc < 2) {
    fatal(__FILE__,__LINE__,"Need input file\n");
  }
  interpretter(argv[1],system,1);

  delete(system);

  MPI_Finalize();
}
 // NYI - debug directive that allows the debugger to attach
