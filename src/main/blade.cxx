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
  void **message; // OMP

#ifdef REPLICAEXCHANGE
  MPI_Init(&argc,&argv);
#endif

  int available;
  int notAvailable=cudaGetDeviceCount(&available);
  if (notAvailable==1) fatal(__FILE__,__LINE__,"No GPUs available\n");
  if (available<omp_get_max_threads()) fatal(__FILE__,__LINE__,"Running with %d omp threads but only %d GPUs\n",omp_get_max_threads(),available);

  message=(void**)calloc(omp_get_max_threads(),sizeof(void*));
#pragma omp parallel
  {
  System *system;

  cudaSetDevice(omp_get_thread_num());

  system=new(System);
  system->id=omp_get_thread_num();
  system->idCount=omp_get_num_threads();
  system->message=message;
  if (system->id!=0) {
    int accessible;
    cudaDeviceCanAccessPeer(&accessible, system->id, 0);
    fprintf(stdout,"Device %d %s access device %d directly\n",system->id,(accessible?"can":"cannot"),0);
    if (accessible) {
      cudaDeviceEnablePeerAccess(0,0); // host 0, required 0
    }
  }
  system->rngGPU=new RngGPU;

  // open input file
  if (argc < 2) {
    fatal(__FILE__,__LINE__,"Need input file\n");
  }
  interpretter(argv[1],system);

  delete(system);
  }
  free(message);

#ifdef REPLICAEXCHANGE
  MPI_Finalize();
#endif
}
