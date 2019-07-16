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

__global__ void cull_blocks_kernel(int3 idDomdec,int3 gridDomdec,int *blockCount,int maxPartnersPerBlock,int *blockPartnerCount,struct DomdecBlockPartners *blockPartners,struct DomdecBlockVolume *blockVolume,real3 box,real rc2)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int domainIdx=(idDomdec.x*gridDomdec.y+idDomdec.y)*gridDomdec.z+idDomdec.z;
  int iBlock=i/32+blockCount[domainIdx]; // 32 threads tag teaming per block
  int partnerDomainIdx;
  int3 idPartnerDomain, idShift;
  char4 shift;
  real3 boxShift;
  int s;
  struct DomdecBlockVolume volume, partnerVolume;
  int j,startBlock,endBlock;
  bool hit;
  int cumHit, passHit;
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

  if (iBlock<blockCount[domainIdx+1]) {
    // (0,0,0) interacts with (1,x,x), (0,1,x), (0,0,1), and (0,0,0), x={-1,0,1}
    partnerPos=0;
    startBlock=i/32; // For first (0,0,0) domain only
    for (idShift.x=0; idShift.x<2; idShift.x++) {
      idPartnerDomain.x=idDomdec.x+idShift.x;
      s=(idPartnerDomain.x==gridDomdec.x?1:0);
      s=(idPartnerDomain.x==-1?-1:s);
      idPartnerDomain.x-=s*gridDomdec.x;
      shift.x=s;
      boxShift.x=s*box.x;
      for (idShift.y=-idShift.x; idShift.y<2; idShift.y++) {
        idPartnerDomain.y=idDomdec.y+idShift.y;
        s=(idPartnerDomain.y==gridDomdec.y?1:0);
        s=(idPartnerDomain.y==-1?-1:s);
        idPartnerDomain.y-=s*gridDomdec.y;
        shift.y=s;
        boxShift.y=s*box.y;
        for (idShift.z=-((idShift.x!=0)|(idShift.y!=0)); idShift.z<2; idShift.z++) {
          idPartnerDomain.z=idDomdec.z+idShift.z;
          s=(idPartnerDomain.z==gridDomdec.z?1:0);
          s=(idPartnerDomain.z==-1?-1:s);
          idPartnerDomain.z-=s*gridDomdec.z;
          shift.z=s;
          boxShift.z=s*box.z;

          partnerDomainIdx=(idPartnerDomain.x*gridDomdec.y+idPartnerDomain.y)*gridDomdec.z+idPartnerDomain.z;
          startBlock+=blockCount[partnerDomainIdx];
          endBlock=blockCount[partnerDomainIdx+1];
          volume=blockVolume[iBlock];
          real3_dec(&volume.max,boxShift);
          real3_dec(&volume.min,boxShift);

          // Check potential partner blocks in groups of 32.
          for (j=startBlock; j<endBlock; j+=32) {
            // Check if this block is interacting
            hit=false;
            if (j+(i&31)<endBlock) {
              partnerVolume=blockVolume[j+(i&31)];
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
              blockPartner.jBlock=j+(i&31);
              blockPartner.shift=shift;
              blockPartner.exclAddress=-1; // No exclusions yet
              // Use i/32 instead of iblock so it's at start of array.
              blockPartners[maxPartnersPerBlock*(i/32)+partnerPos+cumHit-1]=blockPartner;
            }

            // Update partner pos
            __syncwarp();
            partnerPos+=__shfl_sync(0xFFFFFFFF,cumHit,31);
          }

          startBlock=0;
        }
      }
    }

    if ((i&31)==0) {
      // use i/32 instead of iblock so it's at the start of the array
      blockPartnerCount[i/32]=partnerPos;
    }
  }
}

void Domdec::cull_blocks(System *system)
{
  Run *r=system->run;
  int id=system->id-1+(system->idCount==1);
  // int id=(idDomdec.x*gridDomdec.y+idDomdec.y)*gridDomdec.z+idDomdec.z;
  if (id>=0) {
    int localBlockCount=blockCount[id+1]-blockCount[id];
    real rc2=system->run->cutoffs.rCut+cullPad;
    rc2*=rc2;

    cull_blocks_kernel<<<(32*localBlockCount+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(idDomdec,gridDomdec,blockCount_d,maxPartnersPerBlock,blockCandidateCount_d,blockCandidates_d,blockVolume_d,system->state->orthBox,rc2);
  }
}
