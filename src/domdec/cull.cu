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
  if (buffer2>c2) return false;

  bufferB=b.min.y-a.max.y; // Distance one way
  bufferA=a.min.y-b.max.y; // Distance the other way
  bufferA=(bufferA>bufferB?bufferA:bufferB);
  bufferA=(bufferA<0?0:bufferA);
  buffer2+=bufferA*bufferA;
  if (buffer2>c2) return false;

  bufferB=b.min.z-a.max.z; // Distance one way
  bufferA=a.min.z-b.max.z; // Distance the other way
  bufferA=(bufferA>bufferB?bufferA:bufferB);
  bufferA=(bufferA<0?0:bufferA);
  buffer2+=bufferA*bufferA;

  return buffer2<=c2;
}

template <bool flagBox,typename box_type>
__global__ void cull_blocks_kernel(int3 idDomdec,int3 gridDomdec,int *blockCount,int maxPartnersPerBlock,int *blockPartnerCount,struct DomdecBlockPartners *blockPartners,struct DomdecBlockVolume *blockVolume,box_type box,real rc2)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int domainIdx=(idDomdec.x*gridDomdec.y+idDomdec.y)*gridDomdec.z+idDomdec.z;
  int iBlock=i/32+blockCount[domainIdx]; // 32 threads tag teaming per block
  int laneIdx=rectify_modulus(i,32);
  bool iBlockInBounds;
  int partnerDomainIdx;
  int3 idPartnerDomain, idShift;
  char4 shift;
  real3 boxShift;
  int s;
  struct DomdecBlockVolume volume, partnerVolume;
  __shared__ struct DomdecBlockVolume totalVolume, sharedVolume[BLUP>>5], sharedPartnerVolume[32];
  __shared__ int sharedIBlockInBounds[BLUP>>5];
  real3 dist2;
  int jBlock;
  int j,startBlock,endBlock;
  bool hit;
  unsigned int hits;
  int cumHit;
  int partnerPos;
  struct DomdecBlockPartners blockPartner;

// This is the "correct" way to do domain decomposition.
// (0,0,0) domain with (0,0,0) - needs self
// (0,0,0) domain with (1,0,0) - needs self and X
// (0,0,0) domain with (0,1,0) - needs self, X, and Y
// (0,0,0) domain with (1,1,0) - needs self, X, and Y
// (1,0,0) domain with (0,1,0) - needs self, X, and Y
// (0,0,0) domain with (0,0,1) - needs self, X, Y, and Z
// (0,0,0) domain with (1,0,1) - needs self, X, Y, and Z
// (0,0,0) domain with (0,1,1) - needs self, X, Y, and Z
// (0,0,0) domain with (1,1,1) - needs self, X, Y, and Z
// (1,0,0) domain with (0,0,1) - needs self, X, Y, and Z
// (1,0,0) domain with (0,1,1) - needs self, X, Y, and Z
// (0,1,0) domain with (0,0,1) - needs self, X, Y, and Z
// (0,1,0) domain with (1,0,1) - needs self, X, Y, and Z
// (1,1,0) domain with (0,0,1) - needs self, X, Y, and Z
// For the small number of domains here, the naive approach below might be more efficient.

  // Would have an
  // if (iBlock<blockCount[domainIdx+1])
  // wrapping the whole kernel, but some shared memory requires __synchtreads
  iBlockInBounds=(iBlock<blockCount[domainIdx+1]);
  // (0,0,0) interacts with (1,x,x), (0,1,x), (0,0,1), and (0,0,0), x={-1,0,1}
  partnerPos=0;
  int startBlockOffset=i/32; // For first (0,0,0) domain only
  // Load the blockVolume[iBlock] into shared memory for later use
  if (laneIdx==0) {
    sharedIBlockInBounds[threadIdx.x/32]=iBlockInBounds;
    if (iBlockInBounds) {
      sharedVolume[threadIdx.x/32]=blockVolume[iBlock];
    }
  }
  __syncthreads();
  // Find the volume bounding ALL iBlock volumes (one per warp) in this cuda block
  if (threadIdx.x<(BLUP>>5)) {
    volume=sharedVolume[threadIdx.x];
    for (j=1; j<(BLUP>>5); j*=2) {
      partnerVolume.min.x=__shfl_down_sync(0xFF,volume.min.x,j);
      partnerVolume.min.y=__shfl_down_sync(0xFF,volume.min.y,j);
      partnerVolume.min.z=__shfl_down_sync(0xFF,volume.min.z,j);
      partnerVolume.max.x=__shfl_down_sync(0xFF,volume.max.x,j);
      partnerVolume.max.y=__shfl_down_sync(0xFF,volume.max.y,j);
      partnerVolume.max.z=__shfl_down_sync(0xFF,volume.max.z,j);
      if (threadIdx.x+j<(BLUP>>5) && sharedIBlockInBounds[threadIdx.x+j]) {
        volume.min.x=(volume.min.x<partnerVolume.min.x?volume.min.x:partnerVolume.min.x);
        volume.min.y=(volume.min.y<partnerVolume.min.y?volume.min.y:partnerVolume.min.y);
        volume.min.z=(volume.min.z<partnerVolume.min.z?volume.min.z:partnerVolume.min.z);
        volume.max.x=(volume.max.x>partnerVolume.max.x?volume.max.x:partnerVolume.max.x);
        volume.max.y=(volume.max.y>partnerVolume.max.y?volume.max.y:partnerVolume.max.y);
        volume.max.z=(volume.max.z>partnerVolume.max.z?volume.max.z:partnerVolume.max.z);
      }
    }
    if (threadIdx.x==0) {
      totalVolume=volume;
    }
  }
  __syncthreads();
  // Loop over all eligible neighboring domains to find interacting parner blocks
  int y0,x0;
  real partnerMin,partnerWidth;
  for (idShift.z=0; idShift.z<2; idShift.z++) {
    idPartnerDomain.z=idDomdec.z+idShift.z;
    // Check minimum distance to this domain
    dist2.z=0;
    if (idShift.z==1) {
      dist2.z=idPartnerDomain.z*boxzz(box)/gridDomdec.z-totalVolume.max.z;
    }
    dist2.z*=dist2.z;
    // Get periodic shift vector if relevant
    s=(idPartnerDomain.z==gridDomdec.z?1:0);
    // s=(idPartnerDomain.z==-1?-1:s); // SUPERFLUOUS
    idPartnerDomain.z-=s*gridDomdec.z;
    shift.z=s;
    boxShift.z=shift.z*boxzz(box);
    if (flagBox) { // y0 will be 0 if 2*abs(boxzy)<boxyy and gridDomdec.y==1
      y0=-floor(shift.z*boxzy(box)*gridDomdec.y/boxyy(box)+((real)0.5));
    }
    for (idShift.y=-idShift.z; idShift.y<2; idShift.y++) {
      idPartnerDomain.y=idDomdec.y+idShift.y;
      // Check minimum distance to this domain
      if (flagBox) {
        idPartnerDomain.y+=y0;
      }
      partnerWidth=boxyy(box)/gridDomdec.y;
      partnerMin=partnerWidth*idPartnerDomain.y+shift.z*boxzy(box);
      // partnerMax=partnerMin+partnerWidth;
      dist2.y=0;
      if (partnerMin>totalVolume.max.y) {
        dist2.y=partnerMin-totalVolume.max.y;
      } else if (totalVolume.min.y>partnerMin+partnerWidth) {
        dist2.y=totalVolume.min.y-(partnerMin+partnerWidth);
      }
      dist2.y=dist2.z+dist2.y*dist2.y;
      // Get periodic shift vector if relevant
      s=(idPartnerDomain.y>=gridDomdec.y?1:0);
      s=(idPartnerDomain.y<=-1?-1:s);
      if (flagBox) {
        s=(idPartnerDomain.y>=2*gridDomdec.y?2:s);
        s=(idPartnerDomain.y<=-gridDomdec.y-1?-2:s);
      }
      idPartnerDomain.y-=s*gridDomdec.y;
      shift.y=s;
      boxShift.y=shift.z*boxzy(box)+shift.y*boxyy(box);
      if (flagBox) { // y0 will be 0 if 2*abs(boxzy)<boxyy and gridDomdec.y==1
        x0=-floor((shift.z*boxzx(box)+shift.y*boxyx(box))*gridDomdec.x/boxxx(box)+((real)0.5));
      }
      for (idShift.x=-((idShift.z!=0)|(idShift.y!=0)); idShift.x<2; idShift.x++) {
        idPartnerDomain.x=idDomdec.x+idShift.x;
        // Check minimum distance to this domain
        if (flagBox) {
          idPartnerDomain.x+=x0;
        }
        partnerWidth=boxxx(box)/gridDomdec.x;
        partnerMin=partnerWidth*idPartnerDomain.x+(shift.z*boxzx(box)+shift.y*boxyx(box));
        // partnerMax=partnerMin+partnerWidth;
        dist2.x=0;
        if (partnerMin>totalVolume.max.x) {
          dist2.x=partnerMin-totalVolume.max.x;
        } else if (totalVolume.min.x>partnerMin+partnerWidth) {
          dist2.x=totalVolume.min.x-(partnerMin+partnerWidth);
        }
        dist2.x=dist2.y+dist2.x*dist2.x;
        // Get periodic shift vector if relevant
        s=(idPartnerDomain.x==gridDomdec.x?1:0);
        s=(idPartnerDomain.x==-1?-1:s);
        if (flagBox) {
          s=(idPartnerDomain.x>=2*gridDomdec.x?2:s);
          s=(idPartnerDomain.x<=-gridDomdec.x-1?-2:s);
        }
        idPartnerDomain.x-=s*gridDomdec.x;
        shift.x=s;
        boxShift.x=shift.z*boxzx(box)+shift.y*boxyx(box)+shift.x*boxxx(box);

        // Only bother with this domain if it's in range (saves a factor of 3 in one test)
        if (dist2.x<=rc2) {
          partnerDomainIdx=(idPartnerDomain.x*gridDomdec.y+idPartnerDomain.y)*gridDomdec.z+idPartnerDomain.z;
          startBlock=blockCount[partnerDomainIdx];
          endBlock=blockCount[partnerDomainIdx+1];
          // volume=blockVolume[iBlock]; // Old global load
          volume=sharedVolume[threadIdx.x/32]; // New shared load
          real3_dec(&volume.max,boxShift);
          real3_dec(&volume.min,boxShift);

          // Check potential partner blocks in groups of 32.
          for (j=startBlock; j<endBlock; j+=32) {
            jBlock=j+laneIdx;
            // Master warp loads candidate partner volumes into shared memory to save each warp from having to do the same global reads
            __syncthreads();
            if (threadIdx.x<32 && jBlock<endBlock) {
              sharedPartnerVolume[threadIdx.x]=blockVolume[jBlock];
            }
            __syncthreads();

            // Now that shared memory operations are over, we can finally use that boolean flag we set at the beginning of the kernel for control flow
            if (iBlockInBounds) {
              // Check if this block is interacting
              hit=false;
              if (jBlock>=startBlock+startBlockOffset && jBlock<endBlock) {
                // partnerVolume=blockVolume[jBlock]; // Old global load
                partnerVolume=sharedPartnerVolume[laneIdx]; // New shared load
                hit=check_proximity(volume,partnerVolume,rc2);
              }

              // hits=__ballot_sync(0xFFFFFFFF,hit);
              hits=__any_sync(0xFFFFFFFF,hit);
              if (hits) {
                // see how many hits partner threads got
                // Method 1
                // // __syncwarp();
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
                // Method 2 - Using __ballot_sync and __popc seems faster or at least cleaner, but CUDAs left shift doesn't work correctly
                // hits=__ballot_sync(0xFFFFFFFF,hit);
                // cumHit=__popc(hits<<(31-(i&31)));
                // if (cumHit!=__popc(hits<<(31-(i&31)))) printf("hits=0x%08X, hitsShift=0x%08X, cumHit1=%d, cumHit2=%d, lane=%d, hit=%d\n",hits,hits<<(31-(i&31)),cumHit,__popc(hits<<(31-(i&31))),i&31,hit);
                // Method 3
                // hits=__ballot_sync(0xFFFFFFFF,hit);
                // cumHit=__popc(((1<<(i&31))-1)&hits)+hit;
                // if (cumHit!=(__popc(((1<<(i&31))-1)&hits)+hit)) printf("hits=0x%08X, hitsShift=0x%08X, cumHit1=%d, cumHit2=%d, lane=%d, hit=%d, newCumHit=%d, newMask=0x%08X\n",hits,hits<<(31-(i&31)),cumHit,__popc(hits<<(31-(i&31))),i&31,hit,__popc(((1<<(i&31))-1)&hits)+hit,(1<<(i&31))-1);
                // Method 4
                cumHit=hit;
                cumHit+=((i&31)>=1)*__shfl_up_sync(0xFFFFFFFF,cumHit,1);
                cumHit+=((i&31)>=2)*__shfl_up_sync(0xFFFFFFFF,cumHit,2);
                cumHit+=((i&31)>=4)*__shfl_up_sync(0xFFFFFFFF,cumHit,4);
                cumHit+=((i&31)>=8)*__shfl_up_sync(0xFFFFFFFF,cumHit,8);
                cumHit+=((i&31)>=16)*__shfl_up_sync(0xFFFFFFFF,cumHit,16);
                // Method 4 is fastest, followed by method 1, then method 3. Method 2 doesn't work.

                if (hit) {
                  blockPartner.jBlock=jBlock;
                  blockPartner.shift=shift;
                  blockPartner.exclAddress=-1; // No exclusions yet
                  // Use i/32 instead of iblock so it's at start of array.
                  blockPartners[maxPartnersPerBlock*(i/32)+partnerPos+cumHit-1]=blockPartner;
                }

                // Update partner pos
                __syncwarp();
                partnerPos+=__shfl_sync(0xFFFFFFFF,cumHit,31);
              }
            }
          }
        }

        startBlockOffset=0;
      }
    }
  }

  if (iBlockInBounds) {
    if (laneIdx==0) {
      // use i/32 instead of iblock so it's at the start of the array
      blockPartnerCount[i/32]=partnerPos;
      if (partnerPos>=maxPartnersPerBlock) {
// #warning "printf in kernel"
        printf("Error: Overflow of maxPartnersPerBlock. Use \"run setvariable domdecheuristic off\" - except that reallocation is not implemented here\n");
      }
    }
  }
}

template <bool flagBox,typename box_type>
void cull_blocksT(System *system,box_type box)
{
  Run *r=system->run;
  Domdec *d=system->domdec;
  if (d->id>=0) {
    int localBlockCount=d->blockCount[d->id+1]-d->blockCount[d->id];
    real rc2=system->run->cutoffs.rCut+d->cullPad;
    rc2*=rc2;

    cull_blocks_kernel<flagBox><<<(32*localBlockCount+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(d->idDomdec,d->gridDomdec,d->blockCount_d,d->maxPartnersPerBlock,d->blockCandidateCount_d,d->blockCandidates_d,d->blockVolume_d,box,rc2);
  }
}

void Domdec::cull_blocks(System *system)
{
  if (system->state->typeBox) {
    cull_blocksT<true>(system,system->state->tricBox_f);
  } else {
    cull_blocksT<false>(system,system->state->orthBox_f);
  }
}
