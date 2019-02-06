#include <stdio.h>

#include "defines.h"

#ifndef NOMPI
#include <mpi.h>
#endif

void fatalerror(char *message)
{
  fprintf(stderr,"FATAL ERROR: %s\n",message);
  #ifndef NOMPI
  MPI_Finalize();
  #endif
  exit(1);
}

void cudaCheckError(const char fnm[],int line) {
  if (cudaPeekAtLastError() != cudaSuccess) {
    char message[MAXLENGTH];
    sprintf(message,"GPU Error %s, %d: %s\n", fnm,line,cudaGetErrorString(cudaPeekAtLastError()));
    fatalerror(message);
  }
}

