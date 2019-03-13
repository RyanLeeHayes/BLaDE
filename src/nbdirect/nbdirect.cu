#include <cuda_runtime.h>

#include "system/system.h"
#include "msld/msld.h"
#include "system/state.h"
#include "run/run.h"
#include "system/potential.h"
#include "domdec/domdec.h"
#include "main/defines.h"
#include "main/real3.h"



__global__ void getforce_nbdirect_kernel(int blockCount,int *blockBounds,int *blockPartnerCount,struct DomdecBlockPartners *blockPartners,struct NbondPotential *nbonds,int vdwParameterCount,struct VdwPotential *vdwParameters,struct Cutoffs cutoffs,real3 *position,real3 *force,real *lambda,real *lambdaForce,real *energy)
{
// NYI - maybe energy should be a double
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int iBlock=i/32;
  int jBlock;
  int iCount,jCount;
  int ii,jj;
  int j,jmax;
  int ij,jtmp;
  real r,rinv;
  real3 shift;
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
  bool testSelf;

  if (iBlock<blockCount) {
    ii=blockBounds[iBlock];
    iCount=blockBounds[iBlock+1]-ii;
    ii+=(i&31);
    if ((i&31)<iCount) {
      inp=nbonds[ii];
      xi=position[ii];
      bi=inp.siteBlock;
      li=1;
      if (bi) li=lambda[0xFFFF & bi];
    }

    fi=make_float3(0,0,0);
    fli=0;

    jmax=blockPartnerCount[iBlock];
    for (j=0; j<jmax; j++) {
      jBlock=blockPartners[j].jBlock;
      shift=blockPartners[j].shift;
      jj=blockBounds[jBlock];
      jCount=blockBounds[jBlock+1]-jj;
      jj+=(i&31);
      if ((i&31)<jCount) {
        jnp=nbonds[jj];
        xj=position[jj];
        real3_inc(&xj,shift);
        bj=jnp.siteBlock;
        lj=1;
        if (bj) lj=lambda[0xFFFF & bj];
      }
      testSelf=(iBlock==jBlock && shift.x==0 && shift.y==0 && shift.z==0);

      fj=make_float3(0,0,0);
      flj=0;

      for (ij=testSelf; ij<32; ij++) {
        jtmp=i+ij;
        jtmpnp_q=__shfl_sync(0xFFFFFFFF,jnp.q,jtmp);
        int jtmpnp_typeIdx=__shfl_sync(0xFFFFFFFF,jnp.typeIdx,jtmp);
        xjtmp.x=__shfl_sync(0xFFFFFFFF,xj.x,jtmp);
        xjtmp.y=__shfl_sync(0xFFFFFFFF,xj.y,jtmp);
        xjtmp.z=__shfl_sync(0xFFFFFFFF,xj.z,jtmp);
        bjtmp=__shfl_sync(0xFFFFFFFF,bj,jtmp);
        ljtmp=__shfl_sync(0xFFFFFFFF,lj,jtmp);

        fjtmp=make_float3(0,0,0);
        fljtmp=0;
        // if ((i&31)<iCount && (jtmp&31)<jCount)
        jtmp=(testSelf?jtmp:(jtmp&31));
        if ((i&31)<iCount && jtmp<jCount) {
          struct VdwPotential vdwp=vdwParameters[inp.typeIdx*vdwParameterCount+jtmpnp_typeIdx];

          // Geometry
          dr=real3_sub(xi,xjtmp);
          // NOTE #warning "Unprotected sqrt"
          r=real3_mag(dr);

          if (r<cutoffs.rCut) {
            rinv=1/r;

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

            // interaction
              // Electrostatics
            real br=cutoffs.betaEwald*r;
            real erfcrinv=erfcf(br)*rinv;
            fij=-kELECTRIC*inp.q*jtmpnp_q*(erfcrinv-(2/sqrt(M_PI))*cutoffs.betaEwald*expf(-br*br))*rinv;
            if (bi || bj || energy) {
              eij=kELECTRIC*inp.q*jtmpnp_q*erfcrinv;
            }
              // Van der Waals
            real rinv3=rinv*rinv*rinv;
            real rinv6=rinv3*rinv3;
            // fij+=-(12*(vdwp.c12*rinv6)-6*(vdwp.c6))*rinv6*rinv;
            // if (bi || bj || energy) {
            //   eij+=(vdwp.c12*rinv6-vdwp.c6)*rinv6;
            // }
            // fij*=lixljtmp;
            // See charmm/source/domdec/enbxfast.F90, functions calc_vdw_constants, vdw_attraction, vdw_repulsion
            real rCut3=cutoffs.rCut*cutoffs.rCut*cutoffs.rCut;
            real rSwitch3=cutoffs.rSwitch*cutoffs.rSwitch*cutoffs.rSwitch;
            if (r<cutoffs.rSwitch) {
              fij+=(6*vdwp.c6-12*vdwp.c12*rinv6)*rinv6*rinv;
              if (bi || bj || energy) {
                real dv6=-1/(rCut3*rSwitch3);
                real dv12=-dv6*dv6;
                eij+=vdwp.c12*(rinv6*rinv6-dv12)-vdwp.c6*(rinv6-dv6);
              }
            } else {
              real k6=rCut3/(rCut3-rSwitch3);
              real k12=rCut3*rCut3/(rCut3*rCut3-rSwitch3*rSwitch3);
              real rCutinv3=1/rCut3;
              fij+=(6*vdwp.c6*k6*(rinv3-rCutinv3)*rinv3-12*vdwp.c12*k12*(rinv6-rCutinv3*rCutinv3)*rinv6)*rinv;
              if (bi || bj || energy) {
                eij+=vdwp.c12*k12*(rinv6-rCutinv3*rCutinv3)*(rinv6-rCutinv3*rCutinv3)-vdwp.c6*k6*(rinv3-rCutinv3)*(rinv3-rCutinv3);
              }
            }

            // Lambda force
            if ((bi&0xFFFF0000)==(bjtmp&0xFFFF0000)) {
              if (bi==bjtmp) {
                fli+=eij;
                fljtmp=eij;
              } // No else
            } else {
              fli+=ljtmp*eij;
              fljtmp=li*eij;
            }
        
            // Spatial force
            real3_scaleinc(&fi, fij*rinv,dr);
            fjtmp=real3_scale(-fij*rinv,dr);

            // Energy, if requested
            if (energy) {
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
      if ((i&31)<jCount) {
        if (bj) {
          realAtomicAdd(&lambdaForce[0xFFFF & bj],flj);
        }
        at_real3_inc(&force[jj],fj);
      }
    }
    __syncwarp();
    if ((i&31)<iCount) {
      if (bi) {
        realAtomicAdd(&lambdaForce[0xFFFF & bi],fli);
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
  system->domdec->cull_blocks(system);

  Potential *p=system->potential;
  State *s=system->state;
  Msld *m=system->msld;
  Domdec *d=system->domdec;
  int N=d->blockCount;
  int shMem=0;
  real *pEnergy=NULL;

  if (calcEnergy) {
    shMem=BLNB*sizeof(real)/32;
    pEnergy=s->energy_d+eenbdirect;
  }

  getforce_nbdirect_kernel<<<(32*N+BLNB-1)/BLNB,BLNB,shMem,p->nbdirectStream>>>(N,d->blockBounds_d,d->blockPartnerCount_d,d->blockPartners_d,d->localNbonds_d,p->vdwParameterCount,p->vdwParameters_d,system->run->cutoffs,d->localPosition_d,d->localForce_d,m->lambda_d,m->lambdaForce_d,pEnergy);
}

