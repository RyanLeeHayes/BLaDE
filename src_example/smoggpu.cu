
// Written to try out different ion condensation competition models

#include "defines.h"

#include "force.h"
#include "update.h"
#include "md.h"
#include "times.h"
#include "parms.h"
#include "leapparms.h"
#include "state.h"
#include "nblist.h"
#include "files.h"
#include "checkpt.h"
#include "mersenne.h"
#include "adapt.h"
#include "error.h"

#ifndef NOMPI
#include <mpi.h>
#endif

// #include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <signal.h>


#ifdef DEBUG
void arrested_development(void) {
  int i, id=0, nid=1;
  char hostname[200];
  gethostname(hostname,200);
  fprintf(stderr,"PID %d rank %d host %s\n",getpid(),id,hostname);
  raise(SIGSTOP); // Identical behavior, uncatchable.
}
#endif

int main(int argc, char *argv[])
{
  struct_md *md;
  // gmx_cycles_t start,stop;
  gmx_cycles_t start;
  int id=0;
  int gpupernode;

  #ifndef NOMPI
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&id);
  #endif

  #ifdef DEBUG
  arrested_development();
  #endif

  cudaGetDeviceCount(&gpupernode);
  fprintf(stderr,"%d GPUs per device available. Run rank %d on %d.\n",gpupernode,id,id%gpupernode);
  cudaSetDevice(id%gpupernode);
  
  cudaDeviceReset();

  fprintf(stderr,"Beginning initialization\n");
  md=alloc_md(argc,argv);
  readcheckpoint(md);
  uploadkernelargs(md);
  printheader(md);
  fprintf(stderr,"Initialization complete\nBeginning molecular dynamics\n");
  start=gmx_cycles_read();

  do_mt(md,md->state->atoms->mts, 1);
  neighborsearch(md);
  resetforce(md);
  resetenergy(md);
  resetbias(md);

  cudaCheckError(__FILE__,__LINE__);

  while (md->state->step < md->parms->steplim && md->state->walltimeleft) {
    if (md->state->step % md->parms->t_ns == 0) {
      neighborsearch(md);
    }
    getforce(md);
    if (md->state->step % md->parms->t_output == 0)
      printoutput(md);
    if ((md->state->step+1) % md->parms->t_output == 0)
      checkwalltime(md,start);
    if (md->parms->hessian==1) {
      get_FQ(md);
      gethessian(md);
    }
    update(md);
    resetbias(md);
    cudaCheckError(__FILE__,__LINE__);
  }
  fprintf(stderr,"Molecular dynamics complete\nBeginning cleanup\n");

  print_metadynamics_bias(md);
  printsummary(md,start);
  free_md(md);
  fprintf(stderr,"Cleanup complete\n");

  #ifndef NOMPI
  MPI_Finalize();
  #endif
  exit(0);

}
