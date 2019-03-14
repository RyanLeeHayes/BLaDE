#include <cuda_runtime.h>

#include "domdec/domdec.h"
#include "system/system.h"
#include "system/state.h"
#include "run/run.h"
#include "system/potential.h"



Domdec::Domdec()
{
  domain=NULL;
  domain_d=NULL;
  localToGlobal_d=NULL;
  globalToLocal_d=NULL;
  localPosition_d=NULL;
  localForce_d=NULL;
  localNbonds_d=NULL;
  blockSort_d=NULL;
  blockToken_d=NULL;
  blockBounds_d=NULL;
  blockCount=NULL;
  blockCount_d=NULL;
  blockVolume_d=NULL;
  blockPartnerCount_d=NULL;
  blockPartners_d=NULL;
}

Domdec::~Domdec()
{
#warning "No free test"
  free(domain);
  cudaFree(domain_d);
  cudaFree(localToGlobal_d);
  cudaFree(globalToLocal_d);
  cudaFree(localPosition_d);
  cudaFree(localForce_d);
  cudaFree(localNbonds_d);
  cudaFree(blockSort_d);
  cudaFree(blockToken_d);
  cudaFree(blockBounds_d);
  free(blockCount);
  cudaFree(blockCount_d);
  cudaFree(blockVolume_d);
  cudaFree(blockPartnerCount_d);
  cudaFree(blockPartners_d);
}

void Domdec::initialize(System *system)
{
  idDomdec=make_int3(0,0,0);
  if (system->idCount==1) {
    gridDomdec=make_int3(1,1,1);
  } else {
#warning "No 2d or 3d decomposition implemented"
    gridDomdec=make_int3(1,1,system->idCount-1);
    if (system->id!=0) {
      idDomdec=make_int3(0,0,system->id-1);
    }
  }

  int color=(system->idCount==1 || system->id!=0)?0:MPI_UNDEFINED;
  MPI_Comm_split(MPI_COMM_WORLD,color,0,&MPI_COMM_NBOND);

  globalCount=system->state->atomCount;

  // Assume blocks are on average at least 1/3 full, and add some extra blocks for small systems.
  maxBlocks=3*globalCount/32+32;
  real invDensity=(system->state->orthBox.x*system->state->orthBox.y*system->state->orthBox.z)/system->state->atomCount;
  real approxBlockBox=exp(log(32*invDensity)/3);
  real edge=3*approxBlockBox+2*system->run->cutoffs.rCut;
  // edge*edge*edge is the largest volume that can interact with a typically sized box in the worst case. Typically, half these interactions will be taken care of by partner blocks rather than this block, multiplying this expression by 2 means we should have roughly 4 times as many partner spaces as necessary.
  maxPartnersPerBlock=2*((int)(edge*edge*edge/(32*invDensity)));
  fprintf(stdout,"The following parameters are set heuristically at %s:%d, and can cause errors if set too low\n",__FILE__,__LINE__);
  fprintf(stdout,"maxBlocks=%d\n",maxBlocks);
  fprintf(stdout,"maxPartnersPerBlock=%d\n",maxPartnersPerBlock);

  domain=(int*)calloc(globalCount,sizeof(int));
  cudaMalloc(&domain_d,globalCount*sizeof(int));
  cudaMalloc(&localToGlobal_d,globalCount*sizeof(int));
  cudaMalloc(&globalToLocal_d,globalCount*sizeof(int));
  cudaMalloc(&localPosition_d,globalCount*sizeof(real3));
  cudaMalloc(&localForce_d,globalCount*sizeof(real3));
  cudaMalloc(&localNbonds_d,globalCount*sizeof(struct NbondPotential));
  cudaMalloc(&blockSort_d,(globalCount+1)*sizeof(struct DomdecBlockSort));
  cudaMalloc(&blockToken_d,(globalCount+1)*sizeof(struct DomdecBlockToken));
  cudaMalloc(&blockBounds_d,maxBlocks*sizeof(int));
  blockCount=(int*)calloc(gridDomdec.x*gridDomdec.y*gridDomdec.z+1,sizeof(int));
  cudaMalloc(&blockCount_d,(gridDomdec.x*gridDomdec.y*gridDomdec.z+1)*sizeof(int));
  cudaMalloc(&blockVolume_d,maxBlocks*sizeof(struct DomdecBlockVolume));
  cudaMalloc(&blockPartnerCount_d,maxBlocks*sizeof(int));
  cudaMalloc(&blockPartners_d,maxBlocks*maxPartnersPerBlock*sizeof(struct DomdecBlockPartners));

  reset_domdec(system);
}

void Domdec::reset_domdec(System *system)
{
  // Puts each atom in a specific domain/box controlled by one GPU
  assign_domain(system);
  // Splits domains into blocks, or groups of up to 32 nearby atoms
  assign_blocks(system);
}
