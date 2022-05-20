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

template <bool flagBox,typename box_type>
__global__ void recull_blocks_kernel(
  int beginBlock,int endBlock,int maxPartnersPerBlock,
  int *blockCandidateCount,struct DomdecBlockPartners *blockCandidates,
  int *blockPartnerCount,struct DomdecBlockPartners *blockPartners,
  struct DomdecBlockVolume *blockVolume,box_type box,real rc2)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int iBlock=i/32+beginBlock; // 32 threads tag teaming per block
  struct DomdecBlockVolume volume, partnerVolume;
  int j,jmax;
  bool hit;
  // int cumHit, passHit;
  int cumHit;
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
        char4 shift=blockPartner.shift;
        real3 boxShift;
        if (flagBox) {
          boxShift.x=shift.z*boxzx(box)+shift.y*boxyx(box)+shift.x*boxxx(box);
          boxShift.y=shift.z*boxzy(box)+shift.y*boxyy(box);
          boxShift.z=shift.z*boxzz(box);
        } else {
          boxShift.x=shift.x*boxxx(box);
          boxShift.y=shift.y*boxyy(box);
          boxShift.z=shift.z*boxzz(box);
        }
        real3_inc(&partnerVolume.max,boxShift);
        real3_inc(&partnerVolume.min,boxShift);
        hit=check_proximity(volume,partnerVolume,rc2);
      }

      // see how many hits partner threads got
      __syncwarp();
      // cumHit=hit;
      // passHit=((i&1)?0:cumHit); // (i&1) receive
      // cumHit+=__shfl_sync(0xFFFFFFFF,passHit,(i|0)-1);
      // passHit=((i&2)?0:cumHit); // (i&2) receive
      // cumHit+=__shfl_sync(0xFFFFFFFF,passHit,(i|1)-2);
      // passHit=((i&4)?0:cumHit); // (i&4) receive
      // cumHit+=__shfl_sync(0xFFFFFFFF,passHit,(i|3)-4);
      // passHit=((i&8)?0:cumHit); // (i&8) receive
      // cumHit+=__shfl_sync(0xFFFFFFFF,passHit,(i|7)-8);
      // passHit=((i&16)?0:cumHit); // (i&16) receive
      // cumHit+=__shfl_sync(0xFFFFFFFF,passHit,(i|15)-16);
      // Faster cumsum method from cull.cu - no difference here
      cumHit=hit;
      cumHit+=((i&31)>=1)*__shfl_up_sync(0xFFFFFFFF,cumHit,1);
      cumHit+=((i&31)>=2)*__shfl_up_sync(0xFFFFFFFF,cumHit,2);
      cumHit+=((i&31)>=4)*__shfl_up_sync(0xFFFFFFFF,cumHit,4);
      cumHit+=((i&31)>=8)*__shfl_up_sync(0xFFFFFFFF,cumHit,8);
      cumHit+=((i&31)>=16)*__shfl_up_sync(0xFFFFFFFF,cumHit,16);

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

template <bool flagBox,typename box_type>
void recull_blocksT(System *system,box_type box)
{
  Run *r=system->run;
  Domdec *d=system->domdec;
  int beginBlock=d->blockCount[d->id];
  int endBlock=d->blockCount[d->id+1];
  int localBlockCount=endBlock-beginBlock;
  real rc2=system->run->cutoffs.rCut;
  rc2*=rc2;

  recull_blocks_kernel<flagBox><<<(32*localBlockCount+BLUP-1)/BLUP,BLUP,0,r->nbdirectStream>>>(beginBlock,endBlock,d->maxPartnersPerBlock,d->blockCandidateCount_d,d->blockCandidates_d,d->blockPartnerCount_d,d->blockPartners_d,d->blockVolume_d,box,rc2);
}

void Domdec::recull_blocks(System *system)
{
  if (system->state->typeBox) {
    recull_blocksT<true>(system,system->state->tricBox_f);
  } else {
    recull_blocksT<false>(system,system->state->orthBox_f);
  }
}
