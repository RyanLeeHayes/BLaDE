#include <cuda_runtime.h>

#include "domdec/domdec.h"
#include "system/system.h"
#include "system/state.h"
#include "run/run.h"
#include "system/potential.h"
#include "main/defines.h"
#include "main/real3.h"



__host__ __device__ inline
bool check_proximity(DomdecBlockVolume a,DomdecBlockVolume b,real c2)
{
  real bufferA,bufferB,buffer2;

  bufferB=b.min.x-a.max.x; // Distance one way
  bufferA=a.min.x-b.max.x; // Distance the other way
  bufferA=(bufferA>bufferB?bufferA:bufferB);
  bufferA=(bufferA<0?0:bufferA);
  buffer2=bufferA*bufferA;

  bufferB=b.min.y-a.max.y; // Distance one way
  bufferA=a.min.y-b.max.y; // Distance the other way
  bufferA=(bufferA>bufferB?bufferA:bufferB);
  bufferA=(bufferA<0?0:bufferA);
  buffer2+=bufferA*bufferA;

  bufferB=b.min.z-a.max.z; // Distance one way
  bufferA=a.min.z-b.max.z; // Distance the other way
  bufferA=(bufferA>bufferB?bufferA:bufferB);
  bufferA=(bufferA<0?0:bufferA);
  buffer2+=bufferA*bufferA;

  return buffer2<=c2;
}

__global__ void recull_blocks_kernel(
  int beginBlock,int endBlock,int maxPartnersPerBlock,
  int *blockCandidateCount,struct DomdecBlockPartners *blockCandidates,
  int *blockPartnerCount,struct DomdecBlockPartners *blockPartners,
  struct DomdecBlockVolume *blockVolume,real3 box,real rc2)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int iBlock=i/32+beginBlock; // 32 threads tag teaming per block
  struct DomdecBlockVolume volume, partnerVolume;
  int j,jmax;
  bool hit;
  int cumHit, passHit;
  int partnerPos;
  struct DomdecBlockPartners blockPartner;

  if (iBlock<endBlock) {
    partnerPos=0;

    volume=blockVolume[iBlock];
    jmax=blockCandidateCount[i/32];

    // Check potential partner blocks in groups of 32.
    for (j=0; j<jmax; j+=32) {
      // Check if this block is interacting
      hit=false;
      if ((j+(i&31))<jmax) {
        blockPartner=blockCandidates[maxPartnersPerBlock*(i/32)+(j+(i&31))];
        partnerVolume=blockVolume[blockPartner.jBlock];
        real3_inc(&partnerVolume.max,blockPartner.shift);
        real3_inc(&partnerVolume.min,blockPartner.shift);
        hit=check_proximity(volume,partnerVolume,rc2);
      }

      // see how many hits partner threads got
      __syncwarp();
      cumHit=hit;
      passHit=((i&1)?0:cumHit); // (i&1) receive
      cumHit+=__shfl_sync(0xFFFFFFFF,passHit,(i|0)-1);
      passHit=((i&2)?0:cumHit); // (i&2) receive
      cumHit+=__shfl_sync(0xFFFFFFFF,passHit,(i|1)-2);
      passHit=((i&4)?0:cumHit); // (i&4) receive
      cumHit+=__shfl_sync(0xFFFFFFFF,passHit,(i|3)-4);
      passHit=((i&8)?0:cumHit); // (i&8) receive
      cumHit+=__shfl_sync(0xFFFFFFFF,passHit,(i|7)-8);
      passHit=((i&16)?0:cumHit); // (i&16) receive
      cumHit+=__shfl_sync(0xFFFFFFFF,passHit,(i|15)-16);

      if (hit) {
        // Use i/32 instead of iblock so it's at start of array.
        blockPartners[maxPartnersPerBlock*(i/32)+partnerPos+cumHit-1]=blockPartner;
      }

      // Update partner pos
      __syncwarp();
      partnerPos+=__shfl_sync(0xFFFFFFFF,cumHit,31);
    }

    if ((i&31)==0) {
      // use i/32 instead of iblock so it's at the start of the array
      blockPartnerCount[i/32]=partnerPos;
    }
  }
}

void Domdec::recull_blocks(System *system)
{
  Run *r=system->run;
  int id=(idDomdec.x*gridDomdec.y+idDomdec.y)*gridDomdec.z+idDomdec.z;
  int beginBlock=blockCount[id];
  int endBlock=blockCount[id+1];
  int localBlockCount=endBlock-beginBlock;
  real rc2=system->run->cutoffs.rCut;
  rc2*=rc2;

  recull_blocks_kernel<<<(32*localBlockCount+BLUP-1)/BLUP,BLUP,0,r->nbdirectStream>>>(beginBlock,endBlock,maxPartnersPerBlock,blockCandidateCount_d,blockCandidates_d,blockPartnerCount_d,blockPartners_d,blockVolume_d,system->state->orthBox,rc2);
}