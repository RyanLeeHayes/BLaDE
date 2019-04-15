#include <cuda_runtime.h>

#include "system/system.h"
#include "system/state.h"
#include "msld/msld.h"
#include "run/run.h"
#include "system/potential.h"
#include "domdec/domdec.h"
#include "main/defines.h"
#include "main/real3.h"



#define WARPSPERBLOCK 1

template <bool useSoftCore>
__global__ void getforce_nbdirect_kernel(int startBlock,int endBlock,int maxPartnersPerBlock,int *blockBounds,int *blockPartnerCount,struct DomdecBlockPartners *blockPartners,struct NbondPotential *nbonds,int vdwParameterCount,struct VdwPotential *vdwParameters,int *blockExcls,struct Cutoffs cutoffs,real3 *position,real3 *force,real3 box,real *lambda,real *lambdaForce,real *energy)
{
// NYI - maybe energy should be a double
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int iWarp=i>>5; // i/32
  int iThread=i-32*iWarp;
  int iBlock=(iWarp>>WARPSPERBLOCK)+startBlock;
  int jBlock;
  int iCount,jCount;
  int ii,jj;
  int j,jmax;
  int ij,jtmp;
  real r,rinv;
  real3 boxShift;
  real3 dr;
  NbondPotential inp,jnp;
  real jtmpnp_q;
  real fij,eij;
  real lEnergy=0;
  extern __shared__ real sEnergy[];
  real3 xi,xj,xjtmp;
  real3 fi,fj,fjtmp;
  real fli,flj,fljtmp;
  int bi,bj,bjtmp;
  real li,lj,ljtmp,lixljtmp;
  real rEff,dredr,dredll; // Soft core stuff
  int exclMask;

  if (iBlock<endBlock) {
    ii=blockBounds[iBlock];
    iCount=blockBounds[iBlock+1]-ii;
    ii+=(iThread);
    if ((iThread)<iCount) {
      inp=nbonds[ii];
      xi=position[ii];
      bi=inp.siteBlock;
      li=1;
      if (bi) li=lambda[0xFFFF & bi];
    }

    fi=real3_reset();
    fli=0;

    // used i/32 instead of iBlock to shift to beginning of array
    jmax=blockPartnerCount[iWarp>>WARPSPERBLOCK];
    for (j=rectify_modulus(iWarp,1<<WARPSPERBLOCK); j<jmax; j+=(1<<WARPSPERBLOCK)) {
      jBlock=blockPartners[maxPartnersPerBlock*(iWarp>>WARPSPERBLOCK)+j].jBlock;
      boxShift=blockPartners[maxPartnersPerBlock*(iWarp>>WARPSPERBLOCK)+j].shift;
      boxShift.x*=box.x;
      boxShift.y*=box.y;
      boxShift.z*=box.z;
      int exclAddress=blockPartners[maxPartnersPerBlock*(iWarp>>WARPSPERBLOCK)+j].exclAddress;
      if (exclAddress==-1) {
        exclMask=0xFFFFFFFF;
      } else {
        exclMask=blockExcls[32*exclAddress+(iThread)];
      }
      if (iBlock==jBlock && boxShift.x==0 && boxShift.y==0 && boxShift.z==0) {
        exclMask>>=(iThread+1);
        exclMask<<=(iThread+1);
      }
      jj=blockBounds[jBlock];
      jCount=blockBounds[jBlock+1]-jj;
      jj+=(iThread);
      if ((iThread)<jCount) {
        jnp=nbonds[jj];
        xj=position[jj];
        real3_inc(&xj,boxShift);
        bj=jnp.siteBlock;
        lj=1;
        if (bj) lj=lambda[0xFFFF & bj];
      }

      fj=real3_reset();
      flj=0;

      for (ij=0; ij<32; ij++) {
        jtmp=iThread+ij;
        jtmp-=32*(jtmp>=32);
        jtmpnp_q=__shfl_sync(0xFFFFFFFF,jnp.q,jtmp);
        int jtmpnp_typeIdx=__shfl_sync(0xFFFFFFFF,jnp.typeIdx,jtmp);
        xjtmp.x=__shfl_sync(0xFFFFFFFF,xj.x,jtmp);
        xjtmp.y=__shfl_sync(0xFFFFFFFF,xj.y,jtmp);
        xjtmp.z=__shfl_sync(0xFFFFFFFF,xj.z,jtmp);
        bjtmp=__shfl_sync(0xFFFFFFFF,bj,jtmp);
        ljtmp=__shfl_sync(0xFFFFFFFF,lj,jtmp);

        fjtmp=real3_reset();
        fljtmp=0;
        if (iThread<iCount && jtmp<jCount && ((1<<jtmp)&exclMask)) {
          struct VdwPotential vdwp=vdwParameters[inp.typeIdx*vdwParameterCount+jtmpnp_typeIdx];

          // Geometry
          dr=real3_sub(xi,xjtmp);
          // NOTE #warning "Unprotected sqrt"
          r=real3_mag(dr);

          if (r<cutoffs.rCut) {
            // Scaling
            if ((bi&0xFFFF0000)==(bjtmp&0xFFFF0000)) {
              if (bi==bjtmp) {
                lixljtmp=li;
              } else {
                lixljtmp=0;
              }
            } else {
              lixljtmp=li*ljtmp;
            }

            rEff=r;
            if (useSoftCore) {
              dredr=1.0; // d(rEff) / d(r)
              dredll=0.0; // d(rEff) / d(lixljtmp)
              if (bi || bjtmp) {
                real rSoft=(2.0*ANGSTROM*sqrt(4.0))*(1.0-lixljtmp);
                if (r<rSoft) {
                  real rdivrs=r/rSoft;
                  rEff=1.0-0.5*rdivrs;
                  rEff=rEff*rdivrs*rdivrs*rdivrs+0.5;
                  dredr=3.0-2.0*rdivrs;
                  dredr*=rdivrs*rdivrs;
                  dredll=rEff-dredr*rdivrs;
                  dredll*=-(2.0*ANGSTROM*sqrt(4.0));
                  rEff*=rSoft;
                }
              }
            }
            rinv=1/rEff;

            // interaction
              // Electrostatics
            real br=cutoffs.betaEwald*rEff;
            real erfcrinv=erfcf(br)*rinv;
            fij=-kELECTRIC*inp.q*jtmpnp_q*(erfcrinv+(2/sqrt(M_PI))*cutoffs.betaEwald*expf(-br*br))*rinv;
            if (bi || bjtmp || energy) {
              eij=kELECTRIC*inp.q*jtmpnp_q*erfcrinv;
            }
              // Van der Waals
            real rinv3=rinv*rinv*rinv;
            real rinv6=rinv3*rinv3;
            /*fij+=-(12*(vdwp.c12*rinv6)-6*(vdwp.c6))*rinv6*rinv;
            if (bi || bjtmp || energy) {
              eij+=(vdwp.c12*rinv6-vdwp.c6)*rinv6;
            }*/
            // See charmm/source/domdec/enbxfast.F90, functions calc_vdw_constants, vdw_attraction, vdw_repulsion
            real rCut3=cutoffs.rCut*cutoffs.rCut*cutoffs.rCut;
            real rSwitch3=cutoffs.rSwitch*cutoffs.rSwitch*cutoffs.rSwitch;
            if (rEff<cutoffs.rSwitch) {
              fij+=(6*vdwp.c6-12*vdwp.c12*rinv6)*rinv6*rinv;
              if (bi || bjtmp || energy) {
                real dv6=1/(rCut3*rSwitch3);
                eij+=vdwp.c12*(rinv6*rinv6-dv6*dv6)-vdwp.c6*(rinv6-dv6);
              }
            } else {
              real k6=rCut3/(rCut3-rSwitch3);
              real k12=rCut3*rCut3/(rCut3*rCut3-rSwitch3*rSwitch3);
              real rCutinv3=1/rCut3;
              fij+=(6*vdwp.c6*k6*(rinv3-rCutinv3)*rinv3-12*vdwp.c12*k12*(rinv6-rCutinv3*rCutinv3)*rinv6)*rinv;
              if (bi || bjtmp || energy) {
                eij+=vdwp.c12*k12*(rinv6-rCutinv3*rCutinv3)*(rinv6-rCutinv3*rCutinv3)-vdwp.c6*k6*(rinv3-rCutinv3)*(rinv3-rCutinv3);
              }
            }
            fij*=lixljtmp;

            // Lambda force
            if (bi || bjtmp) {
              if (useSoftCore) {
                fljtmp=eij+fij*dredll;
              } else {
                fljtmp=eij;
              }
              if ((bi&0xFFFF0000)==(bjtmp&0xFFFF0000)) {
                if (bi==bjtmp) {
                  fli+=fljtmp;
                } else {
                  fljtmp=0;
                }
              } else {
                fli+=ljtmp*fljtmp;
                fljtmp*=li;
              }
            }
        
            // Spatial force
            if (useSoftCore) {
              rinv=1/r;
              fij*=dredr;
            }
            real3_scaleinc(&fi, fij*rinv,dr);
            fjtmp=real3_scale(-fij*rinv,dr);

            // Energy, if requested
            if (energy) {
              // if (!(lixljtmp*eij>-5000000)) printf("lixljtmp*eij=%f lixljtmp=%f eij=%f\n",lixljtmp*eij,lixljtmp,eij);
              lEnergy+=lixljtmp*eij;
            }
          }
        }
        __syncwarp();
        jtmp=i-ij;
        fj.x+=__shfl_sync(0xFFFFFFFF,fjtmp.x,jtmp);
        fj.y+=__shfl_sync(0xFFFFFFFF,fjtmp.y,jtmp);
        fj.z+=__shfl_sync(0xFFFFFFFF,fjtmp.z,jtmp);
        flj+=__shfl_sync(0xFFFFFFFF,fljtmp,jtmp);
      }
      __syncwarp();
      if ((iThread)<jCount) {
        if (bj) {
          atomicAdd(&lambdaForce[0xFFFF & bj],flj);
        }
        at_real3_inc(&force[jj],fj);
      }
    }
    __syncwarp();
    if ((iThread)<iCount) {
      if (bi) {
        atomicAdd(&lambdaForce[0xFFFF & bi],fli);
      }
      at_real3_inc(&force[ii],fi);
    }
  }

  // Energy, if requested
  if (energy) {
    real_sum_reduce(lEnergy,sEnergy,energy);
  }
}

void getforce_nbdirect(System *system,bool calcEnergy)
{
  system->domdec->pack_positions(system);
  system->domdec->recull_blocks(system);

  Potential *p=system->potential;
  State *s=system->state;
  Run *r=system->run;
  Domdec *d=system->domdec;
  int id=system->id-1+(system->idCount==1);
  int startBlock=d->blockCount[id];
  int endBlock=d->blockCount[id+1];
  int N=endBlock-startBlock;
  int shMem=0;
  real *pEnergy=NULL;

  if (calcEnergy) {
    shMem=BLNB*sizeof(real)/32;
    pEnergy=s->energy_d+eenbdirect;
  }

  if (system->msld->useSoftCore) {
    getforce_nbdirect_kernel<true><<<((32<<WARPSPERBLOCK)*N+BLNB-1)/BLNB,BLNB,shMem,r->nbdirectStream>>>(startBlock,endBlock,d->maxPartnersPerBlock,d->blockBounds_d,d->blockPartnerCount_d,d->blockPartners_d,d->localNbonds_d,p->vdwParameterCount,p->vdwParameters_d,system->domdec->blockExcls_d,system->run->cutoffs,d->localPosition_d,d->localForce_d,s->orthBox,s->lambda_d,s->lambdaForce_d,pEnergy);
  } else {
    getforce_nbdirect_kernel<false><<<((32<<WARPSPERBLOCK)*N+BLNB-1)/BLNB,BLNB,shMem,r->nbdirectStream>>>(startBlock,endBlock,d->maxPartnersPerBlock,d->blockBounds_d,d->blockPartnerCount_d,d->blockPartners_d,d->localNbonds_d,p->vdwParameterCount,p->vdwParameters_d,system->domdec->blockExcls_d,system->run->cutoffs,d->localPosition_d,d->localForce_d,s->orthBox,s->lambda_d,s->lambdaForce_d,pEnergy);
  }

  system->domdec->unpack_forces(system);
}

__global__ void getforce_nbdirect_reduce_kernel(int atomCount,int idCount,real *force)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int j;
  real f=0;

  if (i<atomCount) {
    for (j=1; j<idCount; j++) {
      f+=force[j*atomCount+i];
    }
    atomicAdd(&force[i],f);
  }
}

void getforce_nbdirect_reduce(System *system,bool calcEnergy)
{
  State *s=system->state;
  Run *r=system->run;
  int N=3*s->atomCount+2*s->lambdaCount;

  getforce_nbdirect_reduce_kernel<<<(N+BLNB-1)/BLNB,BLNB,0,r->nbdirectStream>>>(N,system->idCount,s->forceBuffer_d);

  if (calcEnergy) {
    N=eeend;
    getforce_nbdirect_reduce_kernel<<<(N+BLNB-1)/BLNB,BLNB,0,r->nbdirectStream>>>(N,system->idCount,s->energy_d);
  }
}
