#include <cuda_runtime.h>
#include <math.h>

#include "domdec/hbondlist.h"
#include "domdec/domdec.h"
#include "hbond/hbond.h"
#include "main/defines.h"
#include "system/system.h"
#include "system/state.h"
#include "run/run.h"
#include "system/potential.h"
#include "io/io.h"

#include "main/real3.h"

// Step 1: has any watched atom moved more than skin/2 since the last build?
// Every writer writes the same value, so the race on flags[1] is benign.
template <bool flagBox,typename box_type>
__global__ void hbondlist_check_kernel(int watchCount,int *watch,real3 *refPos,real3 *position,box_type box,real halfSkin2,int *flags)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<watchCount) {
    real3 d=real3_subpbc<flagBox>(position[watch[i]],refPos[i],box);
    if (real3_dot<real>(d,d)>halfSkin2) {
      flags[1]=1;
    }
  }
}

// Step 2: zero the entry count before any build thread runs.
// Separate launch, because blocks of the build kernel are not ordered.
__global__ void hbondlist_reset_kernel(int *flags)
{
  if (flags[1]) {
    flags[0]=0; // flags[2] (overflow) is sticky until the host reads it
  }
}

__device__ static inline bool hbond_pair_siteblock(int sD,int sA,int *sb) //helper function to determine blocks for hbondpotential
{
  if ((sD&0xFFFF0000)==(sA&0xFFFF0000)) {   // same site (environment is site 0)
    if (sD!=sA) return false;               // alternative substituents at one site
    sb[0]=sD; sb[1]=0;                      // same block: one lambda (or none, if environment)
  } else if ((0xFFFF&sD)==0) {
    sb[0]=sA; sb[1]=0;
  } else if ((0xFFFF&sA)==0) {
    sb[0]=sD; sb[1]=0;
  } else {
    sb[0]=sD; sb[1]=sA;                     // two different sites: product of lambdas
  }
  return true;
}

// Step 3: brute force over every donor x acceptor pair.
template <bool flagBox,typename box_type>
__global__ void hbondlist_build_kernel(int donorCount,int acceptorCount,HbondDonor *donors,int *acceptors,real3 *position,box_type box,real rList2,int capacity,HbondPotential *list,const struct NbondPotential *nbonds,int *flags)
{
  if (flags[1]==0) return; // no rebuild requested this step (no __syncthreads here, so returning is safe)

  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if (t>=donorCount*acceptorCount) return;

  int d=t/acceptorCount;
  int a=t-d*acceptorCount;
  HbondDonor dn=donors[d];
  int iO=acceptors[a];
  int sb[2];
  real3 dr=real3_subpbc<flagBox>(position[iO],position[dn.idx[0]],box);
  if (real3_dot<real>(dr,dr)<rList2 && hbond_pair_siteblock(dn.siteBlock,nbonds[iO].siteBlock,sb)) {
    int k=atomicAdd(&flags[0],1);
    if (k<capacity) {
      HbondPotential hb;
      hb.idx[0]=iO;        // O
      hb.idx[1]=dn.idx[1]; // H
      hb.idx[2]=dn.idx[0]; // N
      hb.siteBlock[0]=sb[0];
      hb.siteBlock[1]=sb[1];
      list[k]=hb;
    } else {
      flags[2]=1; // overflow: entry dropped, reported to the host
    }
  }
}

// Step 4: remember where the watched atoms were at this build.
__global__ void hbondlist_store_kernel(int watchCount,int *watch,real3 *refPos,real3 *position,int *flags)
{
  if (flags[1]==0) return;
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<watchCount) {
    refPos[i]=position[watch[i]];
  }
}

// Step 5: clear the request. Separate launch, so no thread of step 3 or 4
// can see it cleared early.
__global__ void hbondlist_clear_kernel(int *flags)
{
  flags[1]=0;
}

__global__ void hbondlist_request_kernel(int *flags)
{
  flags[1]=1;
}


// ---------------------------------------------------------------------------
// Domdec builder: scan domdec's candidate block pairs (built with rCut+cullPad
// at the last reset) and keep donor/acceptor pairs with |O-N| < hbondCut+cullPad.
// One warp per local i block, the same layout as cull.cu and nbdirect.cu.
// Candidates form a half list (jBlock>=iBlock within the home image), so both
// orientations are tested for every atom pair, and a block paired with itself
// without a periodic shift only tests lane pairs t>lane, as nbdirect does.
// Positions are read from the global array through localToGlobal, not from
// localPosition_d, which nbdirect repacks concurrently on its own stream.
//
// Block cull: candidates were culled at rCut+cullPad (~21.5 A for HyRes), but
// H-bonds only need hbondCut+cullPad (~8 A). Each warp builds a bounding box of
// the donor N and acceptor O atoms in its i block (once) and in each j block
// (after the periodic shift), and skips the candidate if the boxes are farther
// apart than the list cutoff. The box gap is a lower bound on every atom-pair
// distance, so the cull drops no pair the atom loop would have kept.
// Boxes are built here from the positions already loaded, not read from
// blockVolume_d, which pack_positions rewrites every step on nbdirect's stream.

// Bounding box of the active lanes' positions, returned to every lane.
// Inactive lanes contribute +/-INFINITY. Must be called by the full warp.
__device__ static inline void hbondlist_warp_box(bool active,real3 x,real3 *bmin,real3 *bmax)
{
  real3 mn,mx;
  mn.x=(active?x.x:(real)INFINITY);
  mn.y=(active?x.y:(real)INFINITY);
  mn.z=(active?x.z:(real)INFINITY);
  mx.x=(active?x.x:-(real)INFINITY);
  mx.y=(active?x.y:-(real)INFINITY);
  mx.z=(active?x.z:-(real)INFINITY);
  for (int o=16; o>0; o>>=1) {
    real t;
    t=__shfl_xor_sync(0xFFFFFFFF,mn.x,o); mn.x=(t<mn.x?t:mn.x);
    t=__shfl_xor_sync(0xFFFFFFFF,mn.y,o); mn.y=(t<mn.y?t:mn.y);
    t=__shfl_xor_sync(0xFFFFFFFF,mn.z,o); mn.z=(t<mn.z?t:mn.z);
    t=__shfl_xor_sync(0xFFFFFFFF,mx.x,o); mx.x=(t>mx.x?t:mx.x);
    t=__shfl_xor_sync(0xFFFFFFFF,mx.y,o); mx.y=(t>mx.y?t:mx.y);
    t=__shfl_xor_sync(0xFFFFFFFF,mx.z,o); mx.z=(t>mx.z?t:mx.z);
  }
  *bmin=mn;
  *bmax=mx;
}

// True if the gap between two boxes is within sqrt(r2), as in domdec's check_proximity
__device__ static inline bool hbondlist_boxes_close(real3 aMin,real3 aMax,real3 bMin,real3 bMax,real r2)
{
  real g,g2;
  g=bMin.x-aMax.x; g=(aMin.x-bMax.x>g?aMin.x-bMax.x:g); g=(g<0?0:g); g2=g*g;
  if (g2>r2) return false;
  g=bMin.y-aMax.y; g=(aMin.y-bMax.y>g?aMin.y-bMax.y:g); g=(g<0?0:g); g2+=g*g;
  if (g2>r2) return false;
  g=bMin.z-aMax.z; g=(aMin.z-bMax.z>g?aMin.z-bMax.z:g); g=(g<0?0:g); g2+=g*g;
  return g2<=r2;
}

__device__ static inline void hbondlist_append(int iO,int iN,int sA,int2 range,const HbondDonor *donors,int capacity,HbondPotential *list,int *flags)
{
  int sb[2];
  for (int h=0; h<range.y; h++) {
    if (hbond_pair_siteblock(donors[range.x+h].siteBlock,sA,sb)) {
      int k=atomicAdd(&flags[0],1);
      if (k<capacity) {
        HbondPotential hb;
        hb.idx[0]=iO;                          // O
        hb.idx[1]=donors[range.x+h].idx[1];    // H
        hb.idx[2]=iN;                          // N
        hb.siteBlock[0]=sb[0];
        hb.siteBlock[1]=sb[1];
        list[k]=hb;
      } else {
        flags[2]=1; // overflow, sticky until the host reports it
      }
    }
  }
}

template <bool flagBox,typename box_type>
__global__ void hbondlist_domdec_build_kernel(
  int beginBlock,int endBlock,int maxPartnersPerBlock,
  const int *blockBounds,const int *blockCandidateCount,const struct DomdecBlockPartners *blockCandidates,
  const int *localToGlobal,const int *role,const int2 *donorRange,const HbondDonor *donors,
  const real3 *position,box_type box,real rList2,int capacity,HbondPotential *list,const struct NbondPotential *nbonds,int *flags)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int lane=i&31;
  int iBlock=i/32+beginBlock;
  if (iBlock>=endBlock) return; // uniform across the warp; no __syncthreads below

  int ii=blockBounds[iBlock];
  int iCount=blockBounds[iBlock+1]-ii;
  int gi=-1,ri=0;
  real3 xi;
  xi.x=xi.y=xi.z=0;
  if (lane<iCount) {
    gi=localToGlobal[ii+lane];
    ri=role[gi];
    xi=position[gi];
  }
  unsigned iAcc=__ballot_sync(0xFFFFFFFF,ri&1);
  unsigned iDon=__ballot_sync(0xFFFFFFFF,ri&2);
  if ((iAcc|iDon)==0) return; // block has no backbone donors or acceptors

  // Box around this block's donor N and acceptor O atoms, computed once
  real3 iMin,iMax;
  hbondlist_warp_box(ri!=0,xi,&iMin,&iMax);

  int cmax=blockCandidateCount[i/32];
  for (int c=0; c<cmax; c++) {
    struct DomdecBlockPartners bp=blockCandidates[maxPartnersPerBlock*(i/32)+c];
    int jBlock=bp.jBlock;
    char4 shift=bp.shift;
    int jj=blockBounds[jBlock];
    int jCount=blockBounds[jBlock+1]-jj;
    int gj=-1,rj=0;
    real3 xj;
    xj.x=xj.y=xj.z=0;
    if (lane<jCount) {
      gj=localToGlobal[jj+lane];
      rj=role[gj];
      xj=position[gj];
      if (flagBox) {
        xj.x+=shift.z*boxzx(box)+shift.y*boxyx(box)+shift.x*boxxx(box);
        xj.y+=shift.z*boxzy(box)+shift.y*boxyy(box);
        xj.z+=shift.z*boxzz(box);
      } else {
        xj.x+=shift.x*boxxx(box);
        xj.y+=shift.y*boxyy(box);
        xj.z+=shift.z*boxzz(box);
      }
    }
    unsigned jAcc=__ballot_sync(0xFFFFFFFF,rj&1);
    unsigned jDon=__ballot_sync(0xFFFFFFFF,rj&2);
    if (!((iDon && jAcc) || (jDon && iAcc))) continue; // uniform: nothing to pair

    // Block cull at the H-bond list cutoff (xj already carries the periodic shift)
    real3 jMin,jMax;
    hbondlist_warp_box(rj!=0,xj,&jMin,&jMax);
    if (!hbondlist_boxes_close(iMin,iMax,jMin,jMax,rList2)) continue; // uniform

    bool self=(jBlock==iBlock && shift.x==0 && shift.y==0 && shift.z==0);
    for (int t=0; t<jCount; t++) {
      int gjt=__shfl_sync(0xFFFFFFFF,gj,t);
      int rjt=__shfl_sync(0xFFFFFFFF,rj,t);
      real3 xjt;
      xjt.x=__shfl_sync(0xFFFFFFFF,xj.x,t);
      xjt.y=__shfl_sync(0xFFFFFFFF,xj.y,t);
      xjt.z=__shfl_sync(0xFFFFFFFF,xj.z,t);
      if (lane<iCount && (!self || t>lane)) {
        bool iDonjAcc=(ri&2) && (rjt&1);
        bool jDoniAcc=(rjt&2) && (ri&1);
        if (iDonjAcc || jDoniAcc) {
          real3 d;
          d.x=xi.x-xjt.x;
          d.y=xi.y-xjt.y;
          d.z=xi.z-xjt.z;
          if (d.x*d.x+d.y*d.y+d.z*d.z<rList2) {
            if (iDonjAcc) hbondlist_append(gjt,gi,nbonds[gjt].siteBlock,donorRange[gi],donors,capacity,list,flags);
            if (jDoniAcc) hbondlist_append(gi,gjt,nbonds[gi].siteBlock,donorRange[gjt],donors,capacity,list,flags);
          }
        }
      }
    }
  }
}

template <bool flagBox,typename box_type>
void hbondlist_domdec_updateT(System *system,box_type box)
{
  HbondList *hl=&system->potential->hbondList;
  Domdec *d=system->domdec;
  Run *r=system->run;
  cudaStream_t stream=r->bondedStream;
  
  // Rebuild only when domdec has rebuilt its candidates (or on request)
  if (hl->forceRebuild==false && d->resetSerial==hl->lastResetSerial) return;
  hl->forceRebuild=false;
  hl->lastResetSerial=d->resetSerial;

  int beginBlock=d->blockCount[d->id];
  int endBlock=d->blockCount[d->id+1];
  int localBlockCount=endBlock-beginBlock;
  real rList=r->hbondCut+d->cullPad;

  cudaMemsetAsync(hl->flags_d,0,sizeof(int),stream); // entry count; overflow flag stays sticky
  hbondlist_domdec_build_kernel<flagBox><<<(32*localBlockCount+BLBO-1)/BLBO,BLBO,0,stream>>>(
    beginBlock,endBlock,d->maxPartnersPerBlock,
    d->blockBounds_d,d->blockCandidateCount_d,d->blockCandidates_d,
    d->localToGlobal_d,hl->role_d,hl->donorRange_d,hl->donors_d,
    (real3*)system->state->position_fd,box,rList*rList,hl->capacity,hl->list_d,system->potential->nbonds_d,hl->flags_d);
}

template <bool flagBox,typename box_type>
void hbondlist_updateT(System *system,box_type box)
{
  HbondList *hl=&system->potential->hbondList;
  Run *r=system->run;
  real3 *position=(real3*)system->state->position_fd;
  cudaStream_t stream=r->bondedStream;

  if (hl->forceRebuild) {
    hbondlist_request_kernel<<<1,1,0,stream>>>(hl->flags_d);
    hl->forceRebuild=false;
  }

  hbondlist_check_kernel<flagBox><<<(hl->watchCount+BLBO-1)/BLBO,BLBO,0,stream>>>(hl->watchCount,hl->watch_d,hl->refPos_d,position,box,hl->skin*hl->skin/4,hl->flags_d);
  hbondlist_reset_kernel<<<1,1,0,stream>>>(hl->flags_d);

  int nPair=hl->donorCount*hl->acceptorCount; // checked against INT_MAX at setup
  real rList=r->hbondCut+hl->skin;
  hbondlist_build_kernel<flagBox><<<(nPair+BLBO-1)/BLBO,BLBO,0,stream>>>(hl->donorCount,hl->acceptorCount,hl->donors_d,hl->acceptors_d,position,box,rList*rList,hl->capacity,hl->list_d,system->potential->nbonds_d,hl->flags_d);

  hbondlist_store_kernel<<<(hl->watchCount+BLBO-1)/BLBO,BLBO,0,stream>>>(hl->watchCount,hl->watch_d,hl->refPos_d,position,hl->flags_d);
  hbondlist_clear_kernel<<<1,1,0,stream>>>(hl->flags_d);
}

void hbondlist_update(System *system,bool checkOverflow)
{
  HbondList *hl=&system->potential->hbondList;

  if (system->run->useHbond==false) return;
  if (hl->donorCount==0 || hl->acceptorCount==0) return;

  // The domdec builder sees only this GPU's domain, while bonded terms run on one
  // GPU, so use brute-force builds whenever there is more than one GPU
  bool useDomdec=system->run->hbondListDomdec && system->idCount==1 && system->domdec->id>=0;
  if (useDomdec) {
    if (system->state->typeBox) {
      hbondlist_domdec_updateT<true>(system,system->state->tricBox_f);
    } else {
      hbondlist_domdec_updateT<false>(system,system->state->orthBox_f);
    }
  } else {
    if (system->state->typeBox) {
      hbondlist_updateT<true>(system,system->state->tricBox_f);
    } else {
      hbondlist_updateT<false>(system,system->state->orthBox_f);
    }
  }

  // Overflow means candidates were dropped from the most recent build.
  // Reading the flag syncs the bonded stream, so only do it on energy steps.
  if (checkOverflow) {
    int flags[3];
    cudaMemcpyAsync(flags,hl->flags_d,3*sizeof(int),cudaMemcpyDeviceToHost,system->run->bondedStream);
    cudaStreamSynchronize(system->run->bondedStream);
    if (flags[2]) {
      fatal(__FILE__,__LINE__,"H-bond list overflow: %d candidates for capacity %d. Increase HBONDLIST_PER_DONOR in hbondlist.h, or reduce hbondskin (brute builds).\n",flags[0],hl->capacity);
    }
  }
}
