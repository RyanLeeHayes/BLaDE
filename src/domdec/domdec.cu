#include <cuda_runtime.h>

#include "domdec/domdec.h"
#include "system/system.h"
#include "system/state.h"
#include "run/run.h"
#include "system/potential.h"
#include "holonomic/virtual.h"
#include "main/real3.h"

#ifdef USE_TEXTURE
#include <string.h> // for memset
#endif



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
  blockCandidateCount_d=NULL;
  blockCandidates_d=NULL;
  blockPartnerCount_d=NULL;
  blockPartners_d=NULL;
  localExcls_d=NULL;
  exclSort_d=NULL;
  sortedExcls_d=NULL;
#ifdef USE_TEXTURE
  sortedExcls_tex=0;
#endif
  blockExcls_d=NULL;
  blockExclCount_d=NULL;
}

Domdec::~Domdec()
{
  if (domain) free(domain);
  if (domain_d) cudaFree(domain_d);
  if (localToGlobal_d) cudaFree(localToGlobal_d);
  if (globalToLocal_d) cudaFree(globalToLocal_d);
  if (localPosition_d) cudaFree(localPosition_d);
  if (localForce_d) cudaFree(localForce_d);
  if (localNbonds_d) cudaFree(localNbonds_d);
  if (blockSort_d) cudaFree(blockSort_d);
  if (blockToken_d) cudaFree(blockToken_d);
  if (blockBounds_d) cudaFree(blockBounds_d);
  if (blockCount) free(blockCount);
  if (blockCount_d) cudaFree(blockCount_d);
  if (blockVolume_d) cudaFree(blockVolume_d);
  if (blockCandidateCount_d) cudaFree(blockCandidateCount_d);
  if (blockCandidates_d) cudaFree(blockCandidates_d);
  if (blockPartnerCount_d) cudaFree(blockPartnerCount_d);
  if (blockPartners_d) cudaFree(blockPartners_d);
  if (localExcls_d) cudaFree(localExcls_d);
  if (exclSort_d) cudaFree(exclSort_d);
  if (sortedExcls_d) cudaFree(sortedExcls_d);
#ifdef USE_TEXTURE
  if (sortedExcls_tex) cudaDestroyTextureObject(sortedExcls_tex);
#endif
  if (blockExcls_d) cudaFree(blockExcls_d);
  if (blockExclCount_d) cudaFree(blockExclCount_d);
}

void Domdec::initialize(System *system)
{
  calc_virtual_position(system,true); // Virtual sites / lone pairs
  system->state->broadcast_position(system);
  system->state->broadcast_box(system);

  id=system->id;
  idCount=system->idCount;
  if (!(system->idCount<=2)) {
    id--;
    idCount--;
  }
// #warning "No 2d or 3d decomposition implemented"
  gridDomdec=make_int3(1,1,idCount);
// #warning "If decomposition is performed along any axis other than z, culling will break for non-orthogonal boxes"
  idDomdec=make_int3(0,0,id);

  globalCount=system->state->atomCount;

  // Assume blocks are on average at least 1/3 full, and add some extra blocks for small systems.
  maxBlocks=3*globalCount/32+32;
  // Note: orthBox_f not set yet - now it is, it's set by broadcast_box
  real invDensity;
  if (system->state->typeBox) {
    invDensity=(boxxx(system->state->tricBox_f)*boxyy(system->state->tricBox_f)*boxzz(system->state->tricBox_f))/system->state->atomCount;
  } else {
    invDensity=(boxxx(system->state->orthBox_f)*boxyy(system->state->orthBox_f)*boxzz(system->state->orthBox_f))/system->state->atomCount;
  }
  real approxBlockBox=exp(log(32*invDensity)/3);
  real edge=3*approxBlockBox+2*system->run->cutoffs.rCut;
  // edge*edge*edge is the largest volume that can interact with a typically sized box in the worst case. Typically, half these interactions will be taken care of by partner blocks rather than this block, multiplying this expression by 2 means we should have roughly 4 times as many partner spaces as necessary.
// #warning "Increased maxPartnersPerBlock"
  // maxPartnersPerBlock=2*((int)(edge*edge*edge/(32*invDensity)));
  maxPartnersPerBlock=3*((int)(edge*edge*edge/(32*invDensity)));
  fprintf(stdout,"The following parameters are set heuristically at %s:%d, and can cause errors if set too low\n",__FILE__,__LINE__);
  fprintf(stdout,"maxBlocks=%d\n",maxBlocks);
  fprintf(stdout,"maxPartnersPerBlock=%d\n",maxPartnersPerBlock);

  freqDomdec=10;
  // How far two particles, each with hydrogen/unit mass can get in freqDomdec timesteps, if each has 30 kT of kinetic energy. Incredibly rare to violate this.
  cullPad=2*sqrt(30*kB*system->run->T/1)*freqDomdec*system->run->dt;
  maxBlockExclCount=(4*system->potential->exclCount+1024)/32; // only 32*exclCount is guaranteed, and that only if the box is large. Allocate dynamically
  fprintf(stdout,"freqDomdec=%d (how many steps before domain reset)\n",freqDomdec);
  fprintf(stdout,"cullPad=%g (spatial padding for considering which blocks could interact\n",cullPad);
  fprintf(stdout,"maxBlockExclCount=%d (can reallocate dynamically with \"run setvariable domdecheuristic off\")\n",maxBlockExclCount);

  domain=(int*)calloc(globalCount,sizeof(int));
  cudaMalloc(&domain_d,globalCount*sizeof(int));
  cudaMalloc(&localToGlobal_d,globalCount*sizeof(int));
  cudaMalloc(&globalToLocal_d,globalCount*sizeof(int));
  // #warning "Arbitrarily doubled localPosition_d localForce_d and localNbonds_d size to account for padding. Won't work for small systems. Do something more intelligent."
  // cudaMalloc(&localPosition_d,2*globalCount*sizeof(real3));
  // cudaMalloc(&localForce_d,2*globalCount*sizeof(real3_f));
  // cudaMalloc(&localNbonds_d,2*globalCount*sizeof(struct NbondPotential));
  // Yup, eventually came back to bite me. Doing it right:
  // See also localForce_d size in src/system/potential.cxx
  cudaMalloc(&localPosition_d,32*maxBlocks*sizeof(real3));
  cudaMalloc(&localForce_d,32*maxBlocks*sizeof(real3_f));
  cudaMalloc(&localNbonds_d,32*maxBlocks*sizeof(struct NbondPotential));
  cudaMalloc(&blockSort_d,(globalCount+1)*sizeof(struct DomdecBlockSort));
  cudaMalloc(&blockToken_d,(globalCount+1)*sizeof(struct DomdecBlockToken));
  cudaMalloc(&blockBounds_d,maxBlocks*sizeof(int));
  blockCount=(int*)calloc(idCount+1,sizeof(int));
  cudaMalloc(&blockCount_d,(idCount+1)*sizeof(int));
  cudaMalloc(&blockVolume_d,maxBlocks*sizeof(struct DomdecBlockVolume));
  cudaMalloc(&blockCandidateCount_d,maxBlocks*sizeof(int));
  cudaMalloc(&blockCandidates_d,maxBlocks*maxPartnersPerBlock*sizeof(struct DomdecBlockPartners));
  cudaMalloc(&blockPartnerCount_d,maxBlocks*sizeof(int));
  cudaMalloc(&blockPartners_d,maxBlocks*maxPartnersPerBlock*sizeof(struct DomdecBlockPartners));

  cudaMalloc(&localExcls_d,(system->potential->exclCount+1)*sizeof(struct ExclPotential));
  cudaMalloc(&exclSort_d,(system->potential->exclCount+1)*sizeof(struct DomdecBlockSort));
  cudaMalloc(&sortedExcls_d,system->potential->exclCount*sizeof(struct ExclPotential));
  cudaMalloc(&blockExcls_d,32*maxBlockExclCount*sizeof(int));
  cudaMalloc(&blockExclCount_d,sizeof(int));

#ifdef USE_TEXTURE
  {
    cudaResourceDesc resDesc;
    memset(&resDesc,0,sizeof(resDesc));
    resDesc.resType=cudaResourceTypeLinear;
    resDesc.res.linear.devPtr=sortedExcls_d;
    // pretend it is an int texture instead of int2 texture, because we only ever need to load one element or the other, never both together.
    resDesc.res.linear.desc=cudaCreateChannelDesc<int>();
    resDesc.res.linear.sizeInBytes=2*system->potential->exclCount*sizeof(int);
    cudaTextureDesc texDesc;
    memset(&texDesc,0,sizeof(texDesc));
    texDesc.readMode=cudaReadModeElementType;
    cudaCreateTextureObject(&sortedExcls_tex,&resDesc,&texDesc,NULL);
  }
#endif

  if (system->idCount>0) {
#pragma omp barrier
    system->message[system->id]=(void*)domain_d;
#pragma omp barrier
    domain_omp=(int*)(system->message[0]);
#pragma omp barrier
  }

  reset_domdec(system);
}

void Domdec::reset_domdec(System *system)
{
  // Puts each atom in a specific domain/box controlled by one GPU
  assign_domain(system);
  // Splits domains into blocks, or groups of up to 32 nearby atoms
  assign_blocks(system);
  // Cull blocks to get a candidate list
  cull_blocks(system);
  // Sets up exclusion data structures
  setup_exclusions(system);
}

void Domdec::update_domdec(System *system,bool resetFlag)
{
  calc_virtual_position(system,resetFlag); // Virtual sites / lone pairs
  if (resetFlag) {
    system->domdec->reset_domdec(system);
  } else {
    // Call broadcast_position to call set_fd, even if only one node.
    system->state->broadcast_position(system);
  }
}
