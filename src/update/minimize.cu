#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

#include "system/system.h"
#include "system/state.h"
#include "run/run.h"
#include "system/potential.h"
#include "domdec/domdec.h"
#include "holonomic/holonomic.h"
#include "io/io.h"
#include "main/blade_log.h"

#include "main/real3.h"



__global__ void no_mass_weight_kernel(int N,real* masses,real* ones)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real m;

  if (i<N) {
    m=masses[i];
    ones[i]=(isfinite(m)?1:m);
  }
}

__global__ void sd_acceleration_kernel(int N,struct LeapState ls)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real ism;

  if (i<N) {
    ism=ls.ism[i];
    // No acceleration for massless virtual particles
    ls.v[i]=(isfinite(ism)?-ls.f[i]*ls.ism[i]*ls.ism[i]:0);
  }
}

__global__ void sd_scaling_kernel(int N,struct LeapState ls,real_e *grads2)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real3 grad;
  real grad2=0;
  extern __shared__ real sGrad2[];

  if (i<N) {
    grad.x=ls.v[3*i+0];
    grad.y=ls.v[3*i+1];
    grad.z=ls.v[3*i+2];
    grad2 =grad.x*grad.x;
    grad2+=grad.y*grad.y;
    grad2+=grad.z*grad.z;
  }

  real_sum_reduce(grad2/N,sGrad2,grads2);
  real_max_reduce(grad2,sGrad2,grads2+1);
}

__global__ void sd_position_kernel(int N,struct LeapState ls,real_v *v,real scale,real_x *bx)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real_x x;

  if (i<N) {
    x=ls.x[i];
    if (bx) bx[i]=x;
    ls.x[i]=x+scale*v[i];
  }
}

__global__ void sdfd_dotproduct_kernel(int N,struct LeapState ls,real_v *minDirection,real_e *dot)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real lDot=0;
  extern __shared__ real sDot[];

  if (i<N) {
    lDot=ls.v[i]*minDirection[i];
  }

  real_sum_reduce(3*lDot/N,sDot,dot);
}

void State::min_init(System *system)
{
  // Set masses to 1 for shake during minimization, except virtual sites
  cudaMalloc(&(leapState->ism),(3*atomCount+lambdaCount)*sizeof(real));
  no_mass_weight_kernel<<<(3*atomCount+lambdaCount+BLUP-1)/BLUP,BLUP,
    0,system->run->updateStream>>>
    (3*atomCount+lambdaCount,invsqrtMassBuffer_d,leapState->ism);
  if (system->run->minType==esd || system->run->minType==esdfd) {
    cudaMalloc(&grads2_d,2*sizeof(real_e));
    cudaMemset(grads2_d,0,2*sizeof(real_e));
  }
  if (system->run->minType==esdfd) {
    cudaMalloc(&minDirection_d,3*atomCount*sizeof(real_v));
  }
}

void State::min_dest(System *system)
{
  // Set masses back
  cudaFree(leapState->ism);
  leapState->ism=invsqrtMassBuffer_d;
  if (system->run->minType==esd || system->run->minType==esdfd) {
    cudaFree(grads2_d);
  }
  if (system->run->minType==esdfd) {
    cudaFree(minDirection_d);
  }
}

void State::min_move(int step,int nsteps,System *system)
{
  Run *r=system->run;
  real_e grads2[2];
  real_e currEnergy;
  real_e gradDot[1];
  real scaling, rescaling;
  real frac;

  if (r->minType==esd) {
    if (system->id==0) {
      recv_energy();
      if (system->verbose>1) display_nrg(system);
      currEnergy=energy[eepotential];

      // Check for NaN/Inf energy
      if (!isfinite(currEnergy)) {
        fatal(__FILE__,__LINE__,"BLaDE minimization: energy is NaN or Inf at step %d. Check structure or parameters.",step);
      }

      if (step==0) {
        r->dxRMS=r->dxRMSInit;
      }
      // Adaptive step size: halve first, then conditionally increase
      // Net 1.2x when energy decreases, 0.5x when increases
      r->dxRMS*=0.5;
      if (step>0 && currEnergy<prevEnergy) {
        r->dxRMS*=2.4;
      }
      // Detect stalled minimization (energy unchanged) - apply extra damping
      if (step>0 && currEnergy==prevEnergy) {
        r->dxRMS*=0.5;
        char buf[256];
        snprintf(buf, sizeof(buf), "WARNING: Energy unchanged at step %d, applying extra damping\n", step);
        blade_log(buf);
      }
      // Cap dxRMS to prevent unbounded growth (max 10x initial value)
      if (r->dxRMS > 10.0 * r->dxRMSInit) {
        r->dxRMS = 10.0 * r->dxRMSInit;
      }
      // Compute deltaE BEFORE updating prevEnergy (for MINI> output)
      real_e deltaE = (step > 0) ? currEnergy - prevEnergy : 0.0;
      prevEnergy=currEnergy;
      sd_acceleration_kernel<<<(3*atomCount+BLUP-1)/BLUP,BLUP,
        0,r->updateStream>>>(3*atomCount,*leapState);
      holonomic_velocity(system);
      sd_scaling_kernel<<<(atomCount+BLUP-1)/BLUP,BLUP,
        BLUP*sizeof(real)/32,r->updateStream>>>
        (atomCount,*leapState,grads2_d);
      cudaMemcpy(grads2,grads2_d,2*sizeof(real_e),cudaMemcpyDeviceToHost);
      cudaMemset(grads2_d,0,2*sizeof(real_e));
      real gradRMS = sqrt(grads2[0]);
      real gradMax = sqrt(grads2[1]);

      // Adaptive MINI> output
      if (step % r->nprint == 0 && system->verbose >= 1) {
        char buf[256];
        snprintf(buf, sizeof(buf), "MINI> %6d %14.6f %14.6f %14.6f\n", step, currEnergy, deltaE, gradRMS);
        blade_log(buf);
      }

      // Detailed debug output
      if (system->verbose > 1) {
        char buf[256];
        snprintf(buf, sizeof(buf), "rmsgrad = %f\n", gradRMS);
        blade_log(buf);
        snprintf(buf, sizeof(buf), "maxgrad = %f\n", gradMax);
        blade_log(buf);
      }
      // scaling factor to achieve desired rms displacement
      scaling=r->dxRMS/gradRMS;
      // ratio of allowed maximum displacement over actual maximum displacement
      rescaling=r->dxAtomMax/(scaling*gradMax);
      if (system->verbose > 1) {
        char buf[256];
        snprintf(buf, sizeof(buf), "scaling = %f, rescaling = %f\n", scaling, rescaling);
        blade_log(buf);
      }
      // decrease scaling factor if actual max violates allowed max
      // Only apply rescaling to current step's scaling, not persistently to dxRMS
      // (persistent dxRMS modification causes step size to collapse to zero)
      if (rescaling<1) {
        scaling*=rescaling;
      }
      sd_position_kernel<<<(3*atomCount+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>
        (3*atomCount,*leapState,leapState->v,scaling,positionCons_d);
      holonomic_position(system);
    }
  } else if (r->minType==esdfd) {
    if (system->id==0) {
      recv_energy();
      if (system->verbose>1) display_nrg(system);
      currEnergy=energy[eepotential];

      // Check for NaN/Inf energy
      if (!isfinite(currEnergy)) {
        fatal(__FILE__,__LINE__,"BLaDE minimization (SDFD): energy is NaN or Inf at step %d. Check structure or parameters.",step);
      }

      if (step==0) {
        // Apply initial damping for first step stability
        r->dxRMS=r->dxRMSInit*0.5;
        prevEnergy=currEnergy;
      }
      // Detect stalled minimization (energy unchanged) - apply extra damping
      if (step>0 && currEnergy==prevEnergy) {
        r->dxRMS*=0.5;
        char buf[256];
        snprintf(buf, sizeof(buf), "WARNING (SDFD): Energy unchanged at step %d, applying extra damping\n", step);
        blade_log(buf);
      }
      // Cap dxRMS to prevent unbounded growth (max 10x initial value)
      if (r->dxRMS > 10.0 * r->dxRMSInit) {
        r->dxRMS = 10.0 * r->dxRMSInit;
      }
      // Compute deltaE BEFORE updating prevEnergy (for MINI> output)
      real_e deltaE = (step > 0) ? currEnergy - prevEnergy : 0.0;
      prevEnergy=currEnergy;

      sd_acceleration_kernel<<<(3*atomCount+BLUP-1)/BLUP,BLUP,
        0,r->updateStream>>>(3*atomCount,*leapState);
      holonomic_velocity(system);
      cudaMemcpy(minDirection_d,leapState->v,3*atomCount*sizeof(real_v),cudaMemcpyDeviceToDevice);
      sd_scaling_kernel<<<(atomCount+BLUP-1)/BLUP,BLUP,
        BLUP*sizeof(real)/32,r->updateStream>>>
        (atomCount,*leapState,grads2_d);
      cudaMemcpy(grads2,grads2_d,2*sizeof(real_e),cudaMemcpyDeviceToHost);
      cudaMemset(grads2_d,0,2*sizeof(real_e));
      real gradRMS = sqrt(grads2[0]);
      real gradMax = sqrt(grads2[1]);

      // Adaptive MINI> output
      if (step % r->nprint == 0 && system->verbose >= 1) {
        char buf[256];
        snprintf(buf, sizeof(buf), "MINI> %6d %14.6f %14.6f %14.6f\n", step, currEnergy, deltaE, gradRMS);
        blade_log(buf);
      }

      // Detailed debug output
      if (system->verbose > 1) {
        char buf[256];
        snprintf(buf, sizeof(buf), "rmsgrad = %f\n", gradRMS);
        blade_log(buf);
        snprintf(buf, sizeof(buf), "maxgrad = %f\n", gradMax);
        blade_log(buf);
      }
      // scaling factor to achieve desired rms displacement
      scaling=r->dxRMS/gradRMS;
      // ratio of allowed maximum displacement over actual maximum displacement
      rescaling=r->dxAtomMax/(scaling*gradMax);
      if (system->verbose > 1) {
        char buf[256];
        snprintf(buf, sizeof(buf), "scaling = %f, rescaling = %f\n", scaling, rescaling);
        blade_log(buf);
      }
      // decrease scaling factor if actual max violates allowed max
      // Only apply rescaling to current step's scaling, not persistently to dxRMS
      // (persistent dxRMS modification causes step size to collapse to zero)
      if (rescaling<1) {
        scaling*=rescaling;
      }
      backup_position();
      sd_position_kernel<<<(3*atomCount+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>
        (3*atomCount,*leapState,leapState->v,scaling,positionCons_d);
      holonomic_position(system);
    }
    system->domdec->update_domdec(system,false); // false, no need to update neighbor list
    system->potential->calc_force(0,system); // step 0 to always calculate energy
    if (system->id==0) {
      sd_acceleration_kernel<<<(3*atomCount+BLUP-1)/BLUP,BLUP,
        0,r->updateStream>>>(3*atomCount,*leapState);
      holonomic_velocity(system);
      sdfd_dotproduct_kernel<<<(3*atomCount+BLUP-1)/BLUP,BLUP,
        BLUP*sizeof(real)/32,r->updateStream>>>
        (3*atomCount,*leapState,minDirection_d,grads2_d);
      cudaMemcpy(gradDot,grads2_d,sizeof(real_e),cudaMemcpyDeviceToHost);
      cudaMemset(grads2_d,0,2*sizeof(real_e));
      // grads2[0] is F*F, gradDot is F*Fnew
      frac=grads2[0]/(grads2[0]-gradDot[0]);
      if (system->verbose > 1) {
        char buf[256];
        snprintf(buf, sizeof(buf), "F(x0)*dx = %f, F(x0+dx)*dx = %f, frac = %f\n", grads2[0], gradDot[0], frac);
        blade_log(buf);
      }
      if (frac>1.44 || frac<0) {
        r->dxRMS*=1.2;
      } else if (frac<0.25) {
        r->dxRMS*=0.5;
      } else {
        r->dxRMS*=sqrt(frac);
      }
      // Cap dxRMS to prevent unbounded growth (10x initial value)
      if (r->dxRMS > 10.0 * r->dxRMSInit) {
        r->dxRMS = 10.0 * r->dxRMSInit;
      }
      frac*=(nsteps-step)/(1.0*nsteps);
      if (frac>1 || frac<0) frac=1;
      restore_position();
      sd_position_kernel<<<(3*atomCount+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>
        (3*atomCount,*leapState,minDirection_d,frac*scaling,positionCons_d);
      holonomic_position(system);
    }
  } else {
    fatal(__FILE__,__LINE__,"Error: Unrecognized minimization type %d\n",r->minType);
  }
}
