#include "domdec/domdec.h"
#include "io/io.h"
#include "system/potential.h"
#include "run/run.h"
#include "system/system.h"
#include "main/gpu_check.h"



__host__ __device__ static inline
bool operator<(const ExclPotential& a,const ExclPotential& b)
{
  return (a.idx[0]<b.idx[0] || (a.idx[0]==b.idx[0] &&
         (a.idx[1]<b.idx[1])));
}

__global__ void global_to_local_excl_kernel(int exclCount,struct ExclPotential *excls,int *globalToBlock,int id,int *domain,struct ExclPotential *localExcls)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  struct ExclPotential excl;

  if (i<exclCount+1) {
    if (i<exclCount) {
      excl=excls[i];
      if (domain[excl.idx[0]]==id) {
        excl.idx[0]=globalToBlock[excl.idx[0]];
        excl.idx[1]=globalToBlock[excl.idx[1]];
      } else { // Outside of current domain, don't worry about it, less work during sorting.
        excl.idx[0]=-1;
        excl.idx[1]=-1;
      }
    } else {
      excl.idx[0]=-1;
      excl.idx[1]=-1;
    }
    localExcls[i]=excl;
  }
}

// Modified from assign_blocks_grow_tree_kernel
__global__ void assign_excl_grow_tree_kernel(int tokenCount,struct ExclPotential *tokens,DomdecBlockSort *sort,int *first)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int leafPos=tokenCount; // Root is at end of array
  int nextLeafPos, *nextLeafPosPointer;
  struct ExclPotential token, leafPosToken;
  bool placed=false;

  if (i<tokenCount) {
    token=tokens[i];
    if (token.idx[0]>=0) {
      while (!placed) {
        leafPosToken=tokens[leafPos];
        if (leafPosToken<token) {
          nextLeafPosPointer=&sort[leafPos].upper;
        } else if (token<leafPosToken) {
          nextLeafPosPointer=&sort[leafPos].lower;
        } else {
          first[i]=leafPos;
          return;
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
      first[i]=i;
    } else {
      first[i]=-1;
    }
  }
}

__global__ void assign_excl_count_tree_kernel(int tokenCount,struct ExclPotential *tokens,volatile DomdecBlockSort *sort)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int leafPos, nextLeafPos;
  struct DomdecBlockSort s;
  int count;
  int whoAmI;
  int findWhoAmI;
  bool sister; // boolean for whether a sister exists at a particular level
  bool finished=false;

  if (i<tokenCount) { // MODIFIED
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
          
        if (nextLeafPos==tokenCount) { // Made it all the way up the tree // MODIFIED
          finished=true;
        }

        leafPos=nextLeafPos;
      }
    }
  }
}

// MODIFIED from assign_blocks_localToGlobal_kernel
__global__ void assign_excl_sortedExcls_kernel(int exclCount,int rootPos,struct DomdecBlockSort *sort,struct ExclPotential *localExcls,struct ExclPotential *sortedExcls,int *map)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int leafPos=rootPos;
  int count=0;
  int nextCount;
  bool finished=false;

  if (i<exclCount) {
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
    sortedExcls[i]=localExcls[leafPos];
    map[leafPos]=i;
  }
}

// assign_excl_set_blockExcls_kernel<<<(system->potential->exclCount+BLNB-1)/BLNB,BLNB,0,r->updateStream>>>(system->potential->exclCount,system->potential->excls_d,globalToLane_d,firstExcl_d,mapExcl_d,blockExcls_d);
__global__ void assign_excl_set_blockExcls_kernel(int exclCount,struct ExclPotential *excls,int *globalToLane,int *first,int *map,int *blockExcls)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int ii,jj;
  struct ExclPotential token;
  int unsorted, target;
  int mask;

  if (i<exclCount) {
    unsorted=first[i];
    if (unsorted>=0) {
      target=map[unsorted];
      token=excls[i];
      ii=globalToLane[token.idx[0]];
      jj=globalToLane[token.idx[1]];
      mask=0xFFFFFFFF-(1<<jj); // all ones except a zero in j's position
      atomicAnd(&blockExcls[32*target+ii],mask);
    }
  }
}

__global__ void assign_excl_place_blockExcls_kernel(
  int beginBlock,int endBlock,int *blockBounds,
  int maxPartnersPerBlock,int *blockCandidateCount,
  struct DomdecBlockPartners *blockCandidates,
  int exclCount,
#ifdef USE_TEXTURE
  cudaTextureObject_t sortedExcls
#else
  struct ExclPotential *sortedExcls
#endif
  )
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int iBlock=(i/32)+beginBlock;
  int jBlock;
  int hwidth,probePos,probeBlock;
  int iExclBounds[2];
  int jExclBounds[2];
  int b,j,jmax;
  int exclAddress;

  if (iBlock<endBlock) {
    iExclBounds[0]=0;
    iExclBounds[1]=0;
    {
      for (b=0; b<2; b++) {
        // Binary search for this block

        int lowerPos=-1;
        int upperPos=exclCount;

        // Find half of next highest power of 2 above localCount+1
        hwidth=exclCount; // (localCount+1)-1
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
#ifdef USE_TEXTURE
            probeBlock=tex1Dfetch<int>(sortedExcls,2*probePos);
#else
            probeBlock=sortedExcls[probePos].idx[0];
#endif
            if (probeBlock<iBlock+b) {
              lowerPos=probePos;
            } else {
              upperPos=probePos;
            }
          }
        }
        iExclBounds[b]=upperPos;
      }
    }

    if (iExclBounds[0]==iExclBounds[1]) return; // no exclusions for this block

    jmax=blockCandidateCount[(i/32)];

    for (j=(i&31); j<jmax; j+=32) {

      jBlock=blockCandidates[maxPartnersPerBlock*(i/32)+j].jBlock;
      jExclBounds[0]=0;
      jExclBounds[1]=0;
      {
        for (b=0; b<1; b++) { // only need to get lower bound and see if it matches so b=0 is only pass

          // Binary search for this block

          int lowerPos=iExclBounds[0]-1;
          int upperPos=iExclBounds[1];

          // Find half of next highest power of 2 above localCount+1
          hwidth=upperPos-lowerPos; // (localCount+1)-1
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
#ifdef USE_TEXTURE
              probeBlock=tex1Dfetch<int>(sortedExcls,2*probePos+1);
#else
              probeBlock=sortedExcls[probePos].idx[1];
#endif
              if (probeBlock<jBlock+b) {
                lowerPos=probePos;
              } else {
                upperPos=probePos;
              }
            }
          }
          jExclBounds[b]=upperPos;
        }
      }

#ifdef USE_TEXTURE
      probeBlock=tex1Dfetch<int>(sortedExcls,2*jExclBounds[0]+1);
#else
      probeBlock=sortedExcls[jExclBounds[0]].idx[1];
#endif
      // This requires iExclBounds[0]==iExclBounds[1] return earlier to prevent an error here, otherwise you might get a jBlock match without an iBlock match
      if (probeBlock==jBlock) {
#ifdef USE_TEXTURE
        probeBlock=tex1Dfetch<int>(sortedExcls,2*jExclBounds[0]);
#else
        probeBlock=sortedExcls[jExclBounds[0]].idx[0];
#endif
        if (probeBlock==iBlock) { // double check - small errors in energy if we don't
          exclAddress=jExclBounds[0];
          blockCandidates[maxPartnersPerBlock*(i/32)+j].exclAddress=exclAddress;
        }
      }
    }
  }
}


void Domdec::setup_exclusions(System *system)
{
  if (system->potential->exclCount==0) return;
  Run *r=system->run;

  if (id>=0) {
    // Create localExcls_d (the sort tokens)

    global_to_local_excl_kernel<<<(system->potential->exclCount+BLNB-1)/BLNB,BLNB,0,r->updateStream>>>(system->potential->exclCount,system->potential->excls_d,globalToBlock_d,id,domain_d,localExcls_d);
    gpuCheck(cudaGetLastError());

    // Create exclSort (the sort tree)
    gpuCheck(cudaMemsetAsync(exclSort_d,-1,(system->potential->exclCount+1)*sizeof(struct DomdecBlockSort),r->updateStream));

    assign_excl_grow_tree_kernel<<<(system->potential->exclCount+BLNB-1)/BLNB,BLNB,0,r->updateStream>>>(system->potential->exclCount,localExcls_d,exclSort_d,firstExcl_d);
    gpuCheck(cudaGetLastError());
    assign_excl_count_tree_kernel<<<(system->potential->exclCount+BLNB-1)/BLNB,BLNB,0,r->updateStream>>>(system->potential->exclCount,localExcls_d,exclSort_d);
    gpuCheck(cudaGetLastError());

    // Copy back the sortedExclCount
    gpuCheck(cudaMemcpyAsync(&sortedExclCount,&exclSort_d[system->potential->exclCount].upperCount,sizeof(int),cudaMemcpyDeviceToHost,r->updateStream));

    // If there are too many, make more space
    if (sortedExclCount>maxBlockExclCount) {
      maxBlockExclCount=sortedExclCount+64; // add 64 for a little padding
      if (blockExcls_d) gpuCheck(cudaFree(blockExcls_d));
      gpuCheck(cudaMalloc(&blockExcls_d,32*maxBlockExclCount*sizeof(int)));
    }

    // Create sortedExcl_d (self explanatory, enables binary search for exclusions)

    assign_excl_sortedExcls_kernel<<<(sortedExclCount+BLNB-1)/BLNB,BLNB,0,r->updateStream>>>(sortedExclCount,system->potential->exclCount,exclSort_d,localExcls_d,sortedExcls_d,mapExcl_d);
    gpuCheck(cudaGetLastError());

    // Reset blockExcls_d
    gpuCheck(cudaMemsetAsync(blockExcls_d,-1,32*sortedExclCount*sizeof(int),r->updateStream));

    // Set blockExcls
    assign_excl_set_blockExcls_kernel<<<(system->potential->exclCount+BLNB-1)/BLNB,BLNB,0,r->updateStream>>>(system->potential->exclCount,system->potential->excls_d,globalToLane_d,firstExcl_d,mapExcl_d,blockExcls_d);
    gpuCheck(cudaGetLastError());

    // Use binary search to identify exclusions and load them into the candidate structure

    int beginBlock=blockCount[id];
    int endBlock=blockCount[id+1];
    int localBlockCount=endBlock-beginBlock;
    assign_excl_place_blockExcls_kernel<<<(32*localBlockCount+BLNB-1)/BLNB,BLNB,0,r->updateStream>>>(beginBlock,endBlock,blockBounds_d,maxPartnersPerBlock,blockCandidateCount_d,blockCandidates_d,sortedExclCount,
#ifdef USE_TEXTURE
      sortedExcls_tex
#else
      sortedExcls_d
#endif
      );
    gpuCheck(cudaGetLastError());
  }
}
