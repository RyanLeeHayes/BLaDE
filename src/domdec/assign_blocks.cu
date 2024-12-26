#include <cuda_runtime.h>

#include "domdec/domdec.h"
#include "io/io.h"
#include "system/system.h"
#include "system/state.h"
#include "system/potential.h"
#include "run/run.h"
#include "main/real3.h"



__host__ __device__ static inline
bool operator<(const DomdecBlockToken& a,const DomdecBlockToken& b)
{
  return (a.domain<b.domain || (a.domain==b.domain &&
         (a.ix<b.ix || (a.ix==b.ix &&
         (a.iy<b.iy || (a.iy==b.iy &&
         (a.z<b.z)))))));
}

// assign_blocks_get_tokens_kernel<<<(globalCount+BLUP-1)/BLUP,BLUP,0,system->update->updateStream>>>(globalCount,idDomdec,gridDomdec,domainDiv,domain_d,system->state->position_d,system->state->orthBox,blockToken_d);
__global__ void assign_blocks_get_tokens_kernel(int globalCount,int3 gridDomdec,int2 domainDiv,int *domain,real3 *position,real3 box,struct DomdecBlockToken *tokens)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int3 idDomdec;
  real3 xi;
  real posInDomain;
  struct DomdecBlockToken token;

  if (i<globalCount+1) {
    if (i==globalCount) {
      token.domain=-1;
    } else {
      token.domain=domain[i];
      idDomdec.x=token.domain/(gridDomdec.y*gridDomdec.z);
      idDomdec.y=token.domain/gridDomdec.z-idDomdec.x*gridDomdec.y;
      xi=position[i];
      posInDomain=xi.x*gridDomdec.x/box.x-idDomdec.x;
      token.ix=(int)floor(posInDomain*domainDiv.x);
      // token.ix-=(token.ix>=domainDiv.x?domainDiv.x:0);
      // token.ix+=(token.ix<0?domainDiv.x:0);
      // No need to wrap, already in box. Fudge it in for rounding errors
      token.ix=(token.ix>=domainDiv.x?(domainDiv.x-1):token.ix);
      token.ix=(token.ix<0?0:token.ix);

      posInDomain=xi.y*gridDomdec.y/box.y-idDomdec.y;
      token.iy=(int)floor(posInDomain*domainDiv.y);
      // token.iy-=(token.iy>=domainDiv.y?domainDiv.y:0);
      // token.iy+=(token.iy<0?domainDiv.y:0);
      // No need to wrap, already in box. Fudge it in for rounding errors
      token.iy=(token.iy>=domainDiv.y?(domainDiv.y-1):token.iy);
      token.iy=(token.iy<0?0:token.iy);

      token.z=xi.z;
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
__global__ void assign_blocks_count_tree_kernel(int globalCount,struct DomdecBlockToken *tokens,volatile DomdecBlockSort *sort)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int leafPos, nextLeafPos;
  struct DomdecBlockSort s;
  int count;
  int whoAmI;
  int findWhoAmI;
  bool sister; // boolean for whether a sister exists at a particular level
  bool finished=false;

  if (i<globalCount) {
    s=((struct DomdecBlockSort*)sort)[i];
    // If this is a terminal leaf, start counting up the tree
    if (s.root!=-1 && s.lower==-1 && s.upper==-1) {
      sort[i].lowerCount=0;
      sort[i].upperCount=0;
      leafPos=i;
      while (!finished) {
        nextLeafPos=sort[leafPos].root;
        count=sort[leafPos].lowerCount+sort[leafPos].upperCount+1;
        sister=true;
        
        findWhoAmI=sort[nextLeafPos].lower;
        if (findWhoAmI==-1) {
          sort[nextLeafPos].lowerCount=0;
          sister=false;
        } else if (findWhoAmI==leafPos) {
          sort[nextLeafPos].lowerCount=count;
          whoAmI=0;
        }
        findWhoAmI=sort[nextLeafPos].upper;
        if (findWhoAmI==-1) {
          sort[nextLeafPos].upperCount=0;
          sister=false;
        } else if (findWhoAmI==leafPos) {
          sort[nextLeafPos].upperCount=count;
          whoAmI=1;
        }

        // Try to tell sister to go up tree
        if (sister) {
          if (atomicCAS((int*)&sort[nextLeafPos].whoCounts,-1,1-whoAmI)==-1) { // Succeeded
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

__global__ void assign_blocks_localToGlobal_kernel(int globalCount,struct DomdecBlockSort *sort,int *localToGlobal)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int leafPos=globalCount;
  int count=0;
  int nextCount;
  bool finished=false;

  if (i<globalCount) {
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

// assign_blocks_blockBounds_kernel<<<1,idCount*domainDiv.x*domainDiv.y+1,2*(idCount*domainDiv.x*domainDiv.y+1)*sizeof(int),system->update->updateStream>>>(domainDiv,globalCount,localToGlobal_d,blockToken_d,blockCount_d,blockBounds_d);
// Input
// domainDiv - how many blocks a domain is divided into in the x and y directions
// localCount - entries in localToGlobal
// localToGlobal - list for binary search
// tokens - tokens that were used for making the tree structure. Contain information on which column a particle is in
// Output
// blockCount - pointer to a single int for the total number of blocks
// blockBounds - indices (in the local indexing) of first atom in each block
__global__ void assign_blocks_blockBounds_kernel(int domainCount,int2 domainDiv,int globalCount,int *localToGlobal,struct DomdecBlockToken *tokens,int *blockCount,int *blockBounds,int maxBlocks)
{
  int i,i0;
  int domain;
  int ix,iy;
  int probePos,hwidth,j;
  int blocksInColumn;
  extern __shared__ int columnBounds[]; // Two shared arrays of size blockDim.x+1
  int *cumBlocks=columnBounds+blockDim.x+1; // Two shared arrays of size blockDim.x+1
  struct DomdecBlockToken token,probeToken;

  if (threadIdx.x==0) {
    blockCount[0]=0;
    columnBounds[0]=0;
    cumBlocks[0]=0;
  }

  __syncthreads();

  // Loop through domains (slabs in z direction)
  for (domain=0; domain<domainCount; domain++) {
    // Loops through columns in x and y dimension
    for (i0=0; i0<domainDiv.x*domainDiv.y; i0+=blockDim.x) {
      i=i0+threadIdx.x; // column index
      ix=i/domainDiv.y; // x column index
      iy=i-ix*domainDiv.y; // y column index

      // Create search token
      token.domain=domain;
      token.ix=ix;
      token.iy=iy;
      token.z=INFINITY; // Infinity finds first element in next column

      // Set bounds for binary search
      int lowerPos=-1;
      int upperPos=globalCount;

      // Find half of next highest power of 2 above localCount+1
      hwidth=globalCount; // (localCount+1)-1
      hwidth|=hwidth>>1;
      hwidth|=hwidth>>2;
      hwidth|=hwidth>>4;
      hwidth|=hwidth>>8;
      hwidth|=hwidth>>16;
      hwidth++;
      hwidth=hwidth>>1;

      // Do binary search
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

      // Search complete, save to shared array columnBounds
      int is0=threadIdx.x; // shared memory i position
      int is1=threadIdx.x+1; // shared memory i+1 position
      columnBounds[is1]=upperPos;

      __syncthreads();

      lowerPos=columnBounds[is0];
      blocksInColumn=upperPos-lowerPos;
      blocksInColumn=(blocksInColumn+31)/32;

      // Compute cumulative sum of blocks in all previous columns in shared array cumBlocks
      // Requires shared memory because the number of columns is unrelated to warp size
      cumBlocks[is1]=blocksInColumn;
      for (hwidth=1; hwidth<blockDim.x+1; hwidth*=2) {
        __syncthreads();
        if (hwidth&is1) {
          cumBlocks[is1]+=cumBlocks[(is1|(hwidth-1))-hwidth];
        }
      }

      __syncthreads();

      // Save the block bounds into the blockBounds array
      if (i<domainDiv.x*domainDiv.y) {
        int j0=cumBlocks[is0];
        if (j0+blocksInColumn<=maxBlocks) {
        for (j=0; j<blocksInColumn; j++) {
          blockBounds[j+j0]=lowerPos+32*j;
        }
        } else if (j0<maxBlocks) {
// #warning "printf in kernel, this doesn't affect occupancy of 93.8\% on 2080 TI."
          printf("Error: Overflow of maxBlocks. Use \"run setvariable domdecheuristic off\" - except that reallocation is not implemented here\n");
        }
      }

      __syncthreads();

      // Update columnBounds and cumBlocks for next part of this domain
      if (threadIdx.x==0) {
        columnBounds[0]=columnBounds[blockDim.x];
        cumBlocks[0]=cumBlocks[blockDim.x];
      }
      __syncthreads();
    }

    // Update blockCount for next domain after this one is complete
    if (threadIdx.x==0) {
      blockCount[domain+1]=cumBlocks[0];
    }
    __syncthreads();
  }

  // Set final bound
  if (threadIdx.x==0) {
    // blockBounds[blockCount[domainCount]]=globalCount;
    blockBounds[cumBlocks[0]]=columnBounds[0]; // Equivalent statements
  }
}

__global__ void assign_blocks_localNbonds_kernel(int blockCount,int *blockBounds,int *localToGlobal,int *globalToLocal,NbondPotential *nbonds,NbondPotential *localNbonds)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int iBlock=i/32;
  int iLocal,iGlobal,atomsInBlock;

  if (iBlock<blockCount) {
    iLocal=blockBounds[iBlock];
    atomsInBlock=blockBounds[iBlock+1]-iLocal;
    iLocal+=(i&31);
    if ((i&31)<atomsInBlock) {
      iGlobal=localToGlobal[iLocal];
      globalToLocal[iGlobal]=iLocal;
      localNbonds[i]=nbonds[iGlobal];
    }
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
__global__ void assign_blocks_localPosition_kernel(int blockCount,int *blockBounds,int *localToGlobal,real3 *position,real3 *localPosition,struct DomdecBlockVolume *blockVolume)
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
      localPosition[i]=xi;
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
  Run *r=system->run;

  if (id>=0) { 

    // Get the tokens for sorting

    real3 box;
    if (system->state->typeBox) {
      box.x=system->state->tricBox_f.a.x;
      box.y=system->state->tricBox_f.b.y;
      box.z=system->state->tricBox_f.c.z;
    } else {
      box=system->state->orthBox_f;
    }

    assign_blocks_get_tokens_kernel<<<(globalCount+1+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(globalCount,gridDomdec,domainDiv,domain_d,(real3*)system->state->position_fd,box,blockToken_d);

    // Make the tree structure

    cudaMemsetAsync(blockSort_d,-1,(globalCount+1)*sizeof(struct DomdecBlockSort),r->updateStream);

    assign_blocks_grow_tree_kernel<<<(globalCount+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(globalCount,blockToken_d,blockSort_d);
    assign_blocks_count_tree_kernel<<<(globalCount+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(globalCount,blockToken_d,blockSort_d);

    // Create sorted structures

    assign_blocks_localToGlobal_kernel<<<(globalCount+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(globalCount,blockSort_d,localToGlobal_d);

    int ndiv=domainDiv.x*domainDiv.y;
    ndiv=((ndiv>1024)?1024:ndiv);
    assign_blocks_blockBounds_kernel<<<1,ndiv,2*(ndiv+1)*sizeof(int),r->updateStream>>>(idCount,domainDiv,globalCount,localToGlobal_d,blockToken_d,blockCount_d,blockBounds_d,maxBlocks);

    cudaMemcpy(blockCount,blockCount_d,(idCount+1)*sizeof(int),cudaMemcpyDeviceToHost);

    assign_blocks_localNbonds_kernel<<<(32*blockCount[idCount]+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(blockCount[idCount],blockBounds_d,localToGlobal_d,globalToLocal_d,system->potential->nbonds_d,localNbonds_d);

    // Redundant with pack_positions, needed for call to cull
    assign_blocks_localPosition_kernel<<<(32*blockCount[idCount]+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(blockCount[idCount],blockBounds_d,localToGlobal_d,(real3*)system->state->position_fd,localPosition_d,blockVolume_d);
  }
}

void Domdec::pack_positions(System *system)
{
  Run *r=system->run;
  int N=blockCount[idCount];
  if (id>=0) {
    assign_blocks_localPosition_kernel<<<(32*N+BLUP-1)/BLUP,BLUP,0,r->nbdirectStream>>>(N,blockBounds_d,localToGlobal_d,(real3*)system->state->position_fd,localPosition_d,blockVolume_d);
  }
}

__global__ void unpack_forces_kernel(int blockCount,int *blockBounds,int *localToGlobal,real3_f *force,real3_f *localForce)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int iBlock=i/32;
  int iLocal,atomsInBlock;

  if (iBlock<blockCount) {
    iLocal=blockBounds[iBlock];
    atomsInBlock=blockBounds[iBlock+1]-iLocal;
    iLocal+=(i&31);
    if ((i&31)<atomsInBlock) {
      at_real3_inc(&force[localToGlobal[iLocal]],localForce[i]);
    }
  }
}

void Domdec::unpack_forces(System *system)
{
  Run *r=system->run;
  int N=blockCount[idCount];
  if (id>=0) {
    unpack_forces_kernel<<<(32*N+BLUP-1)/BLUP,BLUP,0,r->nbdirectStream>>>(N,blockBounds_d,localToGlobal_d,(real3_f*)system->state->force_d,localForce_d);
  }
}
