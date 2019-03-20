#include "domdec/domdec.h"
#include "system/potential.h"
#include "update/update.h"
#include "system/system.h"



__host__ __device__ static inline
bool operator<(const ExclPotential& a,const ExclPotential& b)
{
  return (a.idx[0]<b.idx[0] || (a.idx[0]==b.idx[0] &&
         (a.idx[1]<b.idx[1])));
}

__global__ void global_to_local_excl_kernel(int exclCount,struct ExclPotential *excls,int *globalToLocal,int id,int *domain,struct ExclPotential *localExcls)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  struct ExclPotential excl;

  if (i<exclCount+1) {
    if (i<exclCount) {
      excl=excls[i];
      if (domain[excl.idx[0]]==id) {
        excl.idx[0]=globalToLocal[excl.idx[0]];
        excl.idx[1]=globalToLocal[excl.idx[1]];
      } else { // Outside of current domain, don' worry about it, less work during sorting.
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
__global__ void assign_excl_grow_tree_kernel(int tokenCount,struct ExclPotential *tokens,DomdecBlockSort *sort)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int leafPos=tokenCount; // Root is at end of array // MODIFIED
  int nextLeafPos, *nextLeafPosPointer;
  struct ExclPotential token, leafPosToken; // MODIFIED: change data type of token
  bool placed=false;

  if (i<tokenCount) { // MODIFIED: globalCount to more general token count
    token=tokens[i];
    if (token.idx[0]>=0) { // MODIFIED: change how we decide if this data type is interesting
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

__global__ void assign_excl_count_tree_kernel(int tokenCount,struct ExclPotential *tokens,DomdecBlockSort *sort)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int leafPos, nextLeafPos;
  struct DomdecBlockSort s;
  int count;
  int whoAmI;
  bool sister; // boolean for whether a sister exists at a particular level
  bool finished=false;

  if (i<tokenCount) { // MODIFIED
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
          
        if (nextLeafPos==tokenCount) { // Made it all the way up the tree // MODIFIED
          finished=true;
        }

        leafPos=nextLeafPos;
      }
    }
  }
}

// MODIFIED from assign_blocks_localToGlobal_kernel
__global__ void assign_excl_sortedExcls_kernel(int exclCount,int rootPos,struct DomdecBlockSort *sort,struct ExclPotential *localExcls,struct ExclPotential *sortedExcls)
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
  }
}

__global__ void assign_excl_blockExcls_kernel(
  int beginBlock,int endBlock,int *blockBounds,
  int maxPartnersPerBlock,int *blockCandidateCount,
  struct DomdecBlockPartners *blockCandidates,
  int exclCount,struct ExclPotential *sortedExcls,
  int *blockExclCount,int *blockExcls)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int iBlock=(i/32)+beginBlock;
  int jBlock;
  int iBeginAtom=blockBounds[iBlock];
  int iEndAtom=blockBounds[iBlock+1];
  int iAtom=(i&31)+iBeginAtom;
  int jAtom;
  int hwidth,probePos,probeAtom;
  int iExclBounds[2];
  int jExclBounds[2];
  int jBlockAtomBounds[2];
  int b,j,jmax;
  int exclPos;
  int mask;
  int hit;
  int exclAddress;

  if (iBlock<endBlock) {
    iExclBounds[0]=0;
    iExclBounds[1]=0;
    if (iAtom<iEndAtom) {
      for (b=0; b<2; b++) {
        // Binary search for this atom

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
            probeAtom=sortedExcls[probePos].idx[0];
            if (probeAtom<iAtom+b) {
              lowerPos=probePos;
            } else {
              upperPos=probePos;
            }
          }
        }
        iExclBounds[b]=upperPos;
      }
    }

    jmax=blockCandidateCount[(i/32)];

    for (j=0; j<jmax;j++) {
      mask=0xFFFFFFFF;

      jBlock=blockCandidates[maxPartnersPerBlock*(i/32)+j].jBlock;
      jExclBounds[0]=0;
      jExclBounds[1]=0;
      if (iAtom<iEndAtom) {
        for (b=0; b<2; b++) {
          jBlockAtomBounds[b]=blockBounds[jBlock+b];

          // Binary search for this atom

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
              probeAtom=sortedExcls[probePos].idx[1];
              if (probeAtom<jBlockAtomBounds[b]) {
                lowerPos=probePos;
              } else {
                upperPos=probePos;
              }
            }
          }
          jExclBounds[b]=upperPos;
        }

        for (exclPos=jExclBounds[0]; exclPos<jExclBounds[1]; exclPos++) {
          jAtom=sortedExcls[exclPos].idx[1];
          mask^=(1<<(jAtom-jBlockAtomBounds[0]));
        }
      }

      // See if any atoms got hit after that mess of binary searches.
      hit=(mask!=0xFFFFFFFF);
      // Need to put shfl before ||, otherwise short circuit || prevents some shfls from calling which has undefined results.
      hit=(__shfl_xor_sync(0xFFFFFFFF,hit,1) || hit);
      hit=(__shfl_xor_sync(0xFFFFFFFF,hit,2) || hit);
      hit=(__shfl_xor_sync(0xFFFFFFFF,hit,4) || hit);
      hit=(__shfl_xor_sync(0xFFFFFFFF,hit,8) || hit);
      hit=(__shfl_xor_sync(0xFFFFFFFF,hit,16) || hit);

      // If so, try to save exclusions
      if (hit) {
        if ((i&31)==0) {
          // Get an address first
          exclAddress=atomicInc((unsigned int*)blockExclCount,0xFFFFFFFF); // 0xFFFFFFFF increment regardless
          blockCandidates[maxPartnersPerBlock*(i/32)+j].exclAddress=exclAddress;
        }
        exclAddress=__shfl_sync(0xFFFFFFFF,exclAddress,0);
        blockExcls[32*exclAddress+(i&31)]=mask;
      }
    }
  }
}


void Domdec::setup_exclusions(System *system)
{
  if (system->potential->exclCount==0) return;

  int id=system->id-1+(system->idCount==1);

  if (id>=0) {
    // Create localExcls_d (the sort tokens)

    global_to_local_excl_kernel<<<(system->potential->exclCount+BLNB-1)/BLNB,BLNB,0,system->update->updateStream>>>(system->potential->exclCount,system->potential->excls_d,globalToLocal_d,id,domain_d,localExcls_d);

    // Create exclSort (the sort tree)
    cudaMemsetAsync(exclSort_d,-1,(system->potential->exclCount+1)*sizeof(struct DomdecBlockSort),system->update->updateStream);

    assign_excl_grow_tree_kernel<<<(system->potential->exclCount+BLNB-1)/BLNB,BLNB,0,system->update->updateStream>>>(system->potential->exclCount,localExcls_d,exclSort_d);
    assign_excl_count_tree_kernel<<<(system->potential->exclCount+BLNB-1)/BLNB,BLNB,0,system->update->updateStream>>>(system->potential->exclCount,localExcls_d,exclSort_d);

    // Copy back the sortedExclCount

    cudaMemcpyAsync(&sortedExclCount,&exclSort_d[system->potential->exclCount].upperCount,sizeof(int),cudaMemcpyDeviceToHost,system->update->updateStream);

    // Create sortedExcl_d (self explanatory, enables binary search for exclusions)

    assign_excl_sortedExcls_kernel<<<(sortedExclCount+BLNB-1)/BLNB,BLNB,0,system->update->updateStream>>>(sortedExclCount,system->potential->exclCount,exclSort_d,localExcls_d,sortedExcls_d);

    // Use binary search to identify exclusions and load them into the candidate structure

    cudaMemsetAsync(blockExclCount_d,0,sizeof(int),system->update->updateStream);
    int beginBlock=blockCount[id];
    int endBlock=blockCount[id+1];
    int localBlockCount=endBlock-beginBlock;
    assign_excl_blockExcls_kernel<<<(32*localBlockCount+BLNB-1)/BLNB,BLNB,0,system->update->updateStream>>>(beginBlock,endBlock,blockBounds_d,maxPartnersPerBlock,blockCandidateCount_d,blockCandidates_d,sortedExclCount,sortedExcls_d,blockExclCount_d,blockExcls_d);
  }
}
