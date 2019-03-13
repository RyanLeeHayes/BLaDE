#include <cuda_runtime.h>
#include <mpi.h>

#include "domdec/domdec.h"
#include "io/io.h"
#include "system/system.h"
#include "system/state.h"
#include "system/potential.h"
#include "update/update.h"



__host__ __device__ static inline
bool operator<(const DomdecBlockToken& a,const DomdecBlockToken& b)
{
  return (a.ix<b.ix || (a.ix==b.ix &&
         (a.iy<b.iy || (a.iy==b.iy &&
         (a.z<b.z)))));
}

// assign_blocks_get_tokens_kernel<<<(globalCount+BLUP-1)/BLUP,BLUP,0,system->update->updateStream>>>(globalCount,idDomdec,gridDomdec,domainDiv,domain_d,system->state->position_d,system->state->orthBox,blockToken_d);
__global__ void assign_blocks_get_tokens_kernel(int globalCount,int3 idDomdec,int3 gridDomdec,int2 domainDiv,int *domain,real3 *position,real3 box,struct DomdecBlockToken *tokens)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int idDomain=(idDomdec.x*gridDomdec.y+idDomdec.y)*gridDomdec.z+idDomdec.z;
  real3 xi;
  real posInDomain;
  struct DomdecBlockToken token;

  if (i<globalCount+1) {
    if (i<globalCount && domain[i]==idDomain) {
      xi=position[i];
      posInDomain=xi.x*gridDomdec.x/box.x-idDomdec.x;
      token.ix=(int)floor(posInDomain*domainDiv.x);
      token.ix-=(token.ix>=domainDiv.x?domainDiv.x:0);
      token.ix+=(token.ix<0?domainDiv.x:0);
      posInDomain=xi.y*gridDomdec.y/box.y-idDomdec.y;
      token.iy=(int)floor(posInDomain*domainDiv.y);
      token.iy-=(token.iy>=domainDiv.y?domainDiv.y:0);
      token.iy+=(token.iy<0?domainDiv.y:0);
      token.z=xi.z;
    } else { // Label it as a different domain
      token.ix=-1;
    }
    tokens[i]=token;
  }
}

__global__ void assign_blocks_grow_tree_kernel(int globalCount,struct DomdecBlockToken *tokens,DomdecBlockSort *sort)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int leafPos=globalCount; // Root is at end of array
  int nextLeafPos, *nextLeafPosPointer;
  struct DomdecBlockToken token, leafPosToken;
  bool placed=false;

  if (i<globalCount) {
    token=tokens[i];
    if (token.ix>=0) {
      while (!placed) {
        leafPosToken=tokens[leafPos];
        if (leafPosToken<token) {
          nextLeafPosPointer=&sort[leafPos].upper;
        } else {
          nextLeafPosPointer=&sort[leafPos].lower;
        }
        nextLeafPos=*nextLeafPosPointer;
        if (nextLeafPos==-1) { // Try to plant leaf here
          nextLeafPos=atomicCAS(nextLeafPosPointer,-1,i);
          if (nextLeafPos==-1) { // Planting was successful
            placed=true;
            sort[i].root=leafPos;
          }
        }
        if (!placed) {
          leafPos=nextLeafPos;
        }
      }
    }
  }
}

// Work back up the tree to the root, counting leaves
__global__ void assign_blocks_count_tree_kernel(int globalCount,struct DomdecBlockToken *tokens,DomdecBlockSort *sort)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int leafPos, nextLeafPos;
  struct DomdecBlockSort s;
  int count;
  int whoAmI;
  bool sister; // boolean for whether a sister exists at a particular level
  bool finished=false;

  if (i<globalCount) {
    s=sort[i];
    // If this is a terminal leaf, start counting up the tree
    if (s.root!=-1 && s.lower==-1 && s.upper==-1) {
      sort[i].lowerCount=0;
      sort[i].upperCount=0;
      leafPos=i;
      while (!finished) {
        nextLeafPos=sort[leafPos].root;
        count=sort[leafPos].lowerCount+sort[leafPos].upperCount+1;
        sister=true;
        
        if (sort[nextLeafPos].lower==-1) {
          sort[nextLeafPos].lowerCount=0;
          sister=false;
        } else if (sort[nextLeafPos].lower==leafPos) {
          sort[nextLeafPos].lowerCount=count;
          whoAmI=0;
        }
        if (sort[nextLeafPos].upper==-1) {
          sort[nextLeafPos].upperCount=0;
          sister=false;
        } else if (sort[nextLeafPos].upper==leafPos) {
          sort[nextLeafPos].upperCount=count;
          whoAmI=1;
        }

        // Try to tell sister to go up tree
        if (sister) {
          if (atomicCAS(&sort[nextLeafPos].whoCounts,-1,1-whoAmI)==-1) { // Succeeded
            finished=true;
          }
        }
          
        if (nextLeafPos==globalCount) { // Made it all the way up the tree
          finished=true;
        }

        leafPos=nextLeafPos;
      }
    }
  }
}

__global__ void assign_blocks_start_local_kernel(int localCount,int globalCount,struct DomdecBlockSort *sort,int *localToGlobal)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int leafPos=globalCount;
  int count=0;
  int nextCount;
  bool finished=false;

  if (i<localCount) {
    // Move past root, which doesn't represent an actual atom
    leafPos=sort[leafPos].upper;
    while (!finished) {
      nextCount=count+sort[leafPos].lowerCount;
      if (nextCount<i) {
        leafPos=sort[leafPos].upper;
        count=nextCount+1;
      } else if (nextCount>i) {
        leafPos=sort[leafPos].lower;
      } else {
        finished=true;
      }
    }
    localToGlobal[i]=leafPos;
  }
}

// Input
// domainDiv - how many blocks a domain is divided into in the x and y directions
// localCount - entries in localToGlobal
// localToGlobal - list for binary search
// tokens - tokens that were used for making the tree structure. Contain information on which column a particle is in
// Output
// blockCount - pointer to a single int for the total number of blocks
// blockBounds - indices (in the local indexing) of first atom in each block
__global__ void assign_blocks_binary_search_kernel(int2 domainDiv,int localCount,int *localToGlobal,struct DomdecBlockToken *tokens,int *blockCount,int *blockBounds)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int ix=i/domainDiv.x;
  int iy=i-domainDiv.x*ix;
  int lowerPos=-1;
  int upperPos=localCount;
  int probePos,hwidth,j;
  int blocksInColumn;
  extern __shared__ int columnBounds[]; // Two shared arrays of size blockDim.x
  int *cumBlocks=columnBounds+blockDim.x; // Two shared arrays of size blockDim.x
  struct DomdecBlockToken token,probeToken;
  token.ix=ix;
  token.iy=iy;
  token.z=-INFINITY;

  // Find half of next highest power of 2 above localCount+1
  hwidth=localCount; // (localCount+1)-1
  hwidth|=hwidth>>1;
  hwidth|=hwidth>>2;
  hwidth|=hwidth>>4;
  hwidth|=hwidth>>8;
  hwidth|=hwidth>>16;
  hwidth++;
  hwidth=hwidth>>1;

  for (; hwidth>0; hwidth=hwidth>>1) {
    probePos=lowerPos+hwidth;
    if (probePos<upperPos) {
      probeToken=tokens[localToGlobal[probePos]];
      if (probeToken<token) {
        lowerPos=probePos;
      } else {
        upperPos=probePos;
      }
    }
  }

  columnBounds[i]=upperPos;

  __syncthreads();

  if (i<domainDiv.x*domainDiv.y) {
    blocksInColumn=columnBounds[i+1]-columnBounds[i];
    blocksInColumn=(blocksInColumn+31)/32;
    cumBlocks[i+1]=blocksInColumn;
  } else { // if (i==domainDiv.x*domainDiv.y)
    cumBlocks[0]=0; // Someone needs to zero the first entry
    blocksInColumn=1; // Add an entry at the end of block bounds below
  }

  // Requires shared memory because the number of columns is unrelated to warp size
  for (hwidth=1; 2*hwidth<localCount+1; hwidth*=2) {
    __syncthreads();
    if (hwidth&i) {
      cumBlocks[i]+=cumBlocks[(i|(hwidth-1))-hwidth];
    }
  }

  __syncthreads();

  int j0=cumBlocks[i];
  for (j=0; j<blocksInColumn; j++) {
    blockBounds[j+j0]=upperPos+32*j;
  }

  if (i==domainDiv.x*domainDiv.y) {
    blockCount[0]=cumBlocks[domainDiv.x*domainDiv.y];
  }
}

__global__ void assign_blocks_continue_local_kernel(int globalCount,int *localToGlobal,int *globalToLocal,NbondPotential *nbonds,NbondPotential *localNbonds)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<globalCount) {
    int iGlobal=localToGlobal[i];
    globalToLocal[iGlobal]=i;
    localNbonds[i]=nbonds[iGlobal];
  }
}

/* OLD __global__ void assign_blocks_finish_local_kernel(int globalCount,int *localToGlobal,real3 *position,real3 *localPosition)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<globalCount) {
    int iGlobal=localToGlobal[i];
    localPosition[i]=position[iGlobal];
  }
}*/

//    assign_blocks_finish_local_kernel<<<(32*blockCount+BLUP-1)/BLUP,BLUP,0,system->update->updateStream>>>(blockCount,blockBounds_d,localToGlobal_d,(real3*)system->state->position_d,localPosition_d,blockVolume_d);
__global__ void assign_blocks_finish_local_kernel(int blockCount,int *blockBounds,int *localToGlobal,real3 *position,real3 *localPosition,struct DomdecBlockVolume *blockVolume)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int iBlock=i/32;
  int j,iLocal,iGlobal,atomsInBlock;
  real3 xi;
  struct DomdecBlockVolume v,vCompare;

  if (iBlock<blockCount) {
    // Copy over positions to local structure
    iLocal=blockBounds[iBlock];
    atomsInBlock=blockBounds[iBlock+1]-iLocal;
    iLocal+=(i&31);
    if ((i&31)<atomsInBlock) {
      iGlobal=localToGlobal[iLocal];
      xi=position[iGlobal];
      localPosition[iLocal]=xi;
    }

    // Find extreme values
    v.max=xi;
    v.min=xi;
    for (j=1; j<32; j*=2) {
      vCompare.min.x=__shfl_down_sync(0xFFFFFFFF,v.min.x,j);
      vCompare.min.y=__shfl_down_sync(0xFFFFFFFF,v.min.y,j);
      vCompare.min.z=__shfl_down_sync(0xFFFFFFFF,v.min.z,j);
      vCompare.max.x=__shfl_down_sync(0xFFFFFFFF,v.max.x,j);
      vCompare.max.y=__shfl_down_sync(0xFFFFFFFF,v.max.y,j);
      vCompare.max.z=__shfl_down_sync(0xFFFFFFFF,v.max.z,j);
      if ((i&31)+j<atomsInBlock) {
        v.min.x=(v.min.x<vCompare.min.x?v.min.x:vCompare.min.x);
        v.min.y=(v.min.y<vCompare.min.y?v.min.y:vCompare.min.y);
        v.min.z=(v.min.z<vCompare.min.z?v.min.z:vCompare.min.z);
        v.max.x=(v.max.x>vCompare.max.x?v.max.x:vCompare.max.x);
        v.max.y=(v.max.y>vCompare.max.y?v.max.y:vCompare.max.y);
        v.max.z=(v.max.z>vCompare.max.z?v.max.z:vCompare.max.z);
      }
    }
    if ((i&31)==0) {
      blockVolume[iBlock]=v;
    }
  }
}

void Domdec::assign_blocks(System *system)
{
  if (system->idCount==1 || system->id!=0) {

    // Get the tokens for sorting

    real3 box=system->state->orthBox;
    real V=box.x*box.y*box.z;
    real dr=exp(log(32*V/system->state->atomCount)/3); // Target block size
    int2 domainDiv;
    domainDiv.x=(int)ceil(box.x/(dr*gridDomdec.x));
    domainDiv.y=(int)ceil(box.y/(dr*gridDomdec.y));

    assign_blocks_get_tokens_kernel<<<(globalCount+BLUP-1)/BLUP,BLUP,0,system->update->updateStream>>>(globalCount,idDomdec,gridDomdec,domainDiv,domain_d,(real3*)system->state->position_d,box,blockToken_d);

    // Make the tree structure

    cudaMemsetAsync(blockSort_d,-1,(globalCount+1)*sizeof(struct DomdecBlockSort),system->update->updateStream);

    assign_blocks_grow_tree_kernel<<<(globalCount+BLUP-1)/BLUP,BLUP,0,system->update->updateStream>>>(globalCount,blockToken_d,blockSort_d);
    assign_blocks_count_tree_kernel<<<(globalCount+BLUP-1)/BLUP,BLUP,0,system->update->updateStream>>>(globalCount,blockToken_d,blockSort_d);

    // Create sorted structures

    cudaMemcpy(&localCount,&blockSort_d[globalCount].upperCount,sizeof(int),cudaMemcpyDeviceToHost);
    assign_blocks_start_local_kernel<<<(localCount+BLUP-1)/BLUP,BLUP,0,system->update->updateStream>>>(localCount,globalCount,blockSort_d,localToGlobal_d);

    assign_blocks_binary_search_kernel<<<1,domainDiv.x*domainDiv.y+1,2*(domainDiv.x*domainDiv.y+1)*sizeof(int),system->update->updateStream>>>(domainDiv,localCount,localToGlobal_d,blockToken_d,blockCount_d,blockBounds_d);
#warning "Not multidomain compatible"
    cudaMemcpy(&blockCount,blockCount_d,sizeof(int),cudaMemcpyDeviceToHost);
  }

  if (system->idCount>2) {
    int additionalLocalCount;
    int cumulativeLocalCount=localCount;
    int i;

    for (i=0; i<system->idCount-1; i++) {
      additionalLocalCount=localCount;
      MPI_Bcast(&additionalLocalCount,1,MPI_INT,i+1,MPI_COMM_WORLD);
      fatal(__FILE__,__LINE__,"Data is going to all th wrong places on the next line\n");
      MPI_Bcast(&localToGlobal_d[cumulativeLocalCount],additionalLocalCount,MPI_INT,i+1,MPI_COMM_WORLD);
      cumulativeLocalCount+=additionalLocalCount;
    }
  }

  if (system->idCount==1 || system->id!=0) {
    assign_blocks_continue_local_kernel<<<(globalCount+BLUP-1)/BLUP,BLUP,0,system->update->updateStream>>>(globalCount,localToGlobal_d,globalToLocal_d,system->potential->nbonds_d,localNbonds_d);
    // OLD assign_blocks_finish_local_kernel<<<(globalCount+BLUP-1)/BLUP,BLUP,0,system->update->updateStream>>>(globalCount,localToGlobal_d,(real3*)system->state->position_d,localPosition_d);

    // Use pack_positions instead.
    // assign_blocks_finish_local_kernel<<<(32*blockCount+BLUP-1)/BLUP,BLUP,0,system->update->updateStream>>>(blockCount,blockBounds_d,localToGlobal_d,(real3*)system->state->position_d,localPosition_d,blockVolume_d);
  }
}

void Domdec::pack_positions(System *system)
{
  if (system->idCount==1 || system->id!=0) {
    assign_blocks_finish_local_kernel<<<(32*blockCount+BLUP-1)/BLUP,BLUP,0,system->potential->nbdirectStream>>>(blockCount,blockBounds_d,localToGlobal_d,(real3*)system->state->position_d,localPosition_d,blockVolume_d);
  }
}
