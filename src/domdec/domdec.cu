#include <cuda_runtime.h>

#include "domdec/domdec.h"
#include "io/io.h"
#include "system/system.h"
#include "system/state.h"
#include "run/run.h"
#include "system/potential.h"
#include "holonomic/virtual.h"
#include "main/real3.h"

#ifdef USE_TEXTURE
#include <string.h> // for memset
#include "main/gpu_check.h"
#endif



Domdec::Domdec()
{
  domain=NULL;
  domain_d=NULL;
  localToGlobal_d=NULL;
  globalToLane_d=NULL;
  globalToBlock_d=NULL;
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
  firstExcl_d=NULL;
  mapExcl_d=NULL;
  exclSort_d=NULL;
  sortedExcls_d=NULL;
#ifdef USE_TEXTURE
  sortedExcls_tex=0;
#endif
  blockExcls_d=NULL;
  overflowFlag_d=NULL;
}

Domdec::~Domdec()
{
  if (domain) free(domain);
  if (domain_d) gpuCheck(cudaFree(domain_d));
  if (localToGlobal_d) gpuCheck(cudaFree(localToGlobal_d));
  if (globalToLane_d) gpuCheck(cudaFree(globalToLane_d));
  if (globalToBlock_d) gpuCheck(cudaFree(globalToBlock_d));
  if (localPosition_d) gpuCheck(cudaFree(localPosition_d));
  if (localForce_d) gpuCheck(cudaFree(localForce_d));
  if (localNbonds_d) gpuCheck(cudaFree(localNbonds_d));
  if (blockSort_d) gpuCheck(cudaFree(blockSort_d));
  if (blockToken_d) gpuCheck(cudaFree(blockToken_d));
  if (blockBounds_d) gpuCheck(cudaFree(blockBounds_d));
  if (blockCount) free(blockCount);
  if (blockCount_d) gpuCheck(cudaFree(blockCount_d));
  if (blockVolume_d) gpuCheck(cudaFree(blockVolume_d));
  if (blockCandidateCount_d) gpuCheck(cudaFree(blockCandidateCount_d));
  if (blockCandidates_d) gpuCheck(cudaFree(blockCandidates_d));
  if (blockPartnerCount_d) gpuCheck(cudaFree(blockPartnerCount_d));
  if (blockPartners_d) gpuCheck(cudaFree(blockPartners_d));
  if (localExcls_d) gpuCheck(cudaFree(localExcls_d));
  if (firstExcl_d) gpuCheck(cudaFree(firstExcl_d));
  if (mapExcl_d) gpuCheck(cudaFree(mapExcl_d));
  if (exclSort_d) gpuCheck(cudaFree(exclSort_d));
  if (sortedExcls_d) gpuCheck(cudaFree(sortedExcls_d));
#ifdef USE_TEXTURE
  if (sortedExcls_tex) cudaDestroyTextureObject(sortedExcls_tex);
#endif
  if (blockExcls_d) gpuCheck(cudaFree(blockExcls_d));
  if (overflowFlag_d) gpuCheck(cudaFree(overflowFlag_d));
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

  // Set maxBlocks
  // Note: orthBox_f is set by broadcast_box
  real3 box;
  if (system->state->typeBox) {
    box.x=boxxx(system->state->tricBox_f);
    box.y=boxyy(system->state->tricBox_f);
    box.z=boxzz(system->state->tricBox_f);
  } else {
    box.x=boxxx(system->state->orthBox_f);
    box.y=boxyy(system->state->orthBox_f);
    box.z=boxzz(system->state->orthBox_f);
  }
  real invDensity=box.x*box.y*box.z/system->state->atomCount;
  real approxBlockBox=exp(log(32*invDensity)/3);
  domainDiv.x=(int)ceil(box.x/(approxBlockBox*gridDomdec.x));
  domainDiv.y=(int)ceil(box.y/(approxBlockBox*gridDomdec.y));
  // If each column has one empty block at the end, there cannot be more blocks than the number of atoms divided into 32 plus one for each column
  maxBlocks=(globalCount/32+1)+(idCount*domainDiv.x*domainDiv.y);
  if (system->verbose > 0) {
    printlog("maxBlocks=%d\n",maxBlocks);
  }

  // Set maxPartnersPerBlock
  real edge=3*approxBlockBox+2*system->run->cutoffs.rCut;
  // edge*edge*edge is the largest volume that can interact with a typically sized box in the worst case. Typically, half these interactions will be taken care of by partner blocks rather than this block, multiplying this expression by 2 means we should have roughly 4 times as many partner spaces as necessary.
// #warning "Increased maxPartnersPerBlock"
  // maxPartnersPerBlock=2*((int)(edge*edge*edge/(32*invDensity)));
  maxPartnersPerBlock=3*((int)(edge*edge*edge/(32*invDensity)));
  maxPartnersPerBlockLimit=12*maxPartnersPerBlock;
  if (system->verbose > 0) {
    printlog("The following parameters are set heuristically at %s:%d, and can cause errors if set too low\n",__FILE__,__LINE__);
    printlog("maxPartnersPerBlock=%d\n",maxPartnersPerBlock);
  }

  freqDomdec=10;
  // How far two particles, each with hydrogen/unit mass can get in freqDomdec timesteps, if each has 30 kT of kinetic energy. Incredibly rare to violate this.
  cullPad=2*sqrt(30*kB*system->run->T/1)*freqDomdec*system->run->dt;
  maxBlockExclCount=(4*system->potential->exclCount+1024)/32; // only 32*exclCount is guaranteed, and that only if the box is large. Allocate dynamically
  if (system->verbose > 0) {
    printlog("freqDomdec=%d (how many steps before domain reset)\n",freqDomdec);
    printlog("cullPad=%g (spatial padding for considering which blocks could interact\n",cullPad);
    printlog("maxBlockExclCount=%d (will reallocate dynamically)\n",maxBlockExclCount);
  }

  // global count used to be padded to be larger than atomCount so blocks always started at 0, modulus 32. That caused problems.
  domain=(int*)calloc(globalCount,sizeof(int));
  gpuCheck(cudaMalloc(&domain_d,globalCount*sizeof(int)));
  gpuCheck(cudaMalloc(&localToGlobal_d,globalCount*sizeof(int)));
  gpuCheck(cudaMalloc(&globalToLane_d,globalCount*sizeof(int)));
  gpuCheck(cudaMalloc(&globalToBlock_d,globalCount*sizeof(int)));
  gpuCheck(cudaMalloc(&localPosition_d,32*maxBlocks*sizeof(real3)));
  gpuCheck(cudaMalloc(&localForce_d,32*maxBlocks*sizeof(real3_f)));
  gpuCheck(cudaMalloc(&localNbonds_d,32*maxBlocks*sizeof(struct NbondPotential)));
  gpuCheck(cudaMalloc(&blockSort_d,(globalCount+1)*sizeof(struct DomdecBlockSort)));
  gpuCheck(cudaMalloc(&blockToken_d,(globalCount+1)*sizeof(struct DomdecBlockToken)));
  gpuCheck(cudaMalloc(&blockBounds_d,maxBlocks*sizeof(int)));
  blockCount=(int*)calloc(idCount+1,sizeof(int));
  gpuCheck(cudaMalloc(&blockCount_d,(idCount+1)*sizeof(int)));
  gpuCheck(cudaMalloc(&blockVolume_d,maxBlocks*sizeof(struct DomdecBlockVolume)));
  gpuCheck(cudaMalloc(&blockCandidateCount_d,maxBlocks*sizeof(int)));
  gpuCheck(cudaMalloc(&blockCandidates_d,maxBlocks*maxPartnersPerBlock*sizeof(struct DomdecBlockPartners)));
  gpuCheck(cudaMalloc(&blockPartnerCount_d,maxBlocks*sizeof(int)));
  gpuCheck(cudaMalloc(&blockPartners_d,maxBlocks*maxPartnersPerBlock*sizeof(struct DomdecBlockPartners)));

  gpuCheck(cudaMalloc(&localExcls_d,(system->potential->exclCount+1)*sizeof(struct ExclPotential)));
  gpuCheck(cudaMalloc(&firstExcl_d,(system->potential->exclCount+1)*sizeof(int)));
  gpuCheck(cudaMalloc(&mapExcl_d,(system->potential->exclCount+1)*sizeof(int)));
  gpuCheck(cudaMalloc(&exclSort_d,(system->potential->exclCount+1)*sizeof(struct DomdecBlockSort)));
  gpuCheck(cudaMalloc(&sortedExcls_d,system->potential->exclCount*sizeof(struct ExclPotential)));
  gpuCheck(cudaMalloc(&blockExcls_d,32*maxBlockExclCount*sizeof(int)));
  gpuCheck(cudaMalloc(&overflowFlag_d,sizeof(int)));

#ifdef USE_TEXTURE
  if (system->potential->exclCount>0) {
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
