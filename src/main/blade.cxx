#include <omp.h>
#include <cuda_runtime.h>

#include "system/system.h"
#include "rng/rng_gpu.h"
#include "io/io.h"

#ifdef REPLICAEXCHANGE
#include <mpi.h>
#endif

int main(int argc, char *argv[])
{
  System *system;

#ifdef REPLICAEXCHANGE
  MPI_Init(&argc,&argv);
#endif

  system=init_system(0,NULL);

#pragma omp parallel
  {
    // open input file
    if (argc < 2) {
      fatal(__FILE__,__LINE__,"Need input file\n");
    }
    interpretter(argv[1],&system[omp_get_thread_num()]);
  }

  dest_system(system);

#ifdef REPLICAEXCHANGE
  MPI_Finalize();
#endif
}
