#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

#include "system/system.h"
#include "system/state.h"
#include "run/run.h"
#include "system/potential.h"
#include "domdec/domdec.h"
#include "holonomic/holonomic.h"
#include "holonomic/rectify.h"
#include "io/io.h"

#include "main/real3.h"

#include <functional>
#include "msld/msld.h"
#include "main/gpu_check.h"



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

template <typename real_type_src, typename real_type_dst>
__global__ void type_conversion_copy(int N, real_type_src* buffer1, real_type_dst* buffer2)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<N) {
    buffer2[i] = (real_type_dst)buffer1[i];
  }
}

void State::min_init(System *system)
{
  // Set masses to 1 for shake during minimization, except virtual sites
  gpuCheck(cudaMalloc(&(leapState->ism),(3*atomCount+lambdaCount)*sizeof(real)));
  no_mass_weight_kernel<<<(3*atomCount+lambdaCount+BLUP-1)/BLUP,BLUP,
    0,system->run->updateStream>>>
    (3*atomCount+lambdaCount,invsqrtMassBuffer_d,leapState->ism);
  gpuCheck(cudaGetLastError());
  if (system->run->minType==esd || system->run->minType==esdfd) {
    gpuCheck(cudaMalloc(&grads2_d,2*sizeof(real_e)));
    gpuCheck(cudaMemset(grads2_d,0,2*sizeof(real_e)));
  }
  if (system->run->minType==esdfd) {
    gpuCheck(cudaMalloc(&minDirection_d,3*atomCount*sizeof(real_v)));
  }
  if (system->run->minType==elbfgs){
    Potential* p = system->potential;
    int sum = p->triangleConsCount + p->branch1ConsCount + p->branch2ConsCount + p->branch3ConsCount;
    if(sum > 0){
      printlog("L-BFGS Check: Total SHAKE Constraints = %d\n", sum);
      fatal(__FILE__,__LINE__,"Error, cannot minimize using L-BFGS with constraints. Ensure \"structure shake none\" is in your minimize file.\n",
      sum);
    }
    // Function called by L-BFGS class to get energy & gradient 
    std::function<real_x()> energy_and_grad = [system](){
      // Potential & Grad Eval -> step=0 to calc energy
      system->domdec->update_domdec(system,true); // domdec and updates single precision array with double precision values
      system->potential->calc_force(0, system);
      // grad(F(X)) G already stored
      system->state->recv_energy();
      // Copy float array written by BLaDE onto double used by L-BFGS
      int DOF = system->state->atomCount*3;
      int shift = system->state->lambdaCount;
      type_conversion_copy<real, real_x><<<(DOF+BLUP-1)/BLUP,BLUP,0,system->run->updateStream>>>(
            DOF, system->state->forceBuffer_d+shift, system->state->forceBufferX_d+shift);
      gpuCheck(cudaGetLastError());
      system->run->lbfgs_energy_evals++;
      return system->state->energy[eepotential];
    };

    int shift = system->state->lambdaCount;
    int DOF = system->state->atomCount*3; // only minimize atom positions
    system->run->lbfgs = new LBFGS(system->run->lbfgs_m, system->run->lbfgs_eps, DOF, system->verbose,
      energy_and_grad, system->state->positionBuffer_d+shift, system->state->forceBufferX_d+shift);
  }
}

void State::min_dest(System *system)
{
  // Set masses back
  gpuCheck(cudaFree(leapState->ism));
  leapState->ism=invsqrtMassBuffer_d;
  if (system->run->minType==esd || system->run->minType==esdfd) {
    gpuCheck(cudaFree(grads2_d));
  }
  if (system->run->minType==esdfd) {
    gpuCheck(cudaFree(minDirection_d));
  }
  if (system->run->minType==elbfgs){
    system->run->lbfgs->~LBFGS();
  }
}

void State::min_move(int step,int nsteps,System *system)
{
  Run *r=system->run;
  real_e grads2[2], gradRMS, gradMax;
  real_e currEnergy, deltaE;
  real_e gradDot[1];
  real scaling, rescaling;
  real frac;

  if (system->id==0) {
    recv_energy();
    if (system->verbose!=0) display_nrg(system);
    currEnergy=energy[eepotential];
    // Check for NaN/Inf energy
    if (!isfinite(currEnergy)) {
      fatal(__FILE__,__LINE__,"BLaDE minimization: energy is NaN or Inf at step %d. Check structure or parameters.",step);
    }
    // Compute deltaE BEFORE updating prevEnergy (for MINI> output)
    deltaE = (step > 0) ? currEnergy - prevEnergy : 0.0;
  }

  if (r->minType==elbfgs){
    if (system->id==0){
      r->lbfgs->minimize_step(currEnergy);
      if(r->lbfgs->minimized){ 
        r->step = nsteps; 
        if (system->verbose) {
          printlog("%d force evaluations wasted in outer loop.\n", step);
          printlog("L-BFGS (m=%d) did %d force evaluations in %d steps. Reset memory %d times.\n", 
            r->lbfgs->m, r->lbfgs_energy_evals, r->lbfgs->step_count, r->lbfgs->reset_count);
          printlog("U0: %f, Uf: %f, Uf - U0: %f\n", r->lbfgs->U0, r->lbfgs->Uf, r->lbfgs->Uf - r->lbfgs->U0);
        }
      } 
      prevEnergy=currEnergy;
      gradRMS=r->lbfgs->rmsg;
    }
  } else if (r->minType==esd) {
    if (system->id==0) {
      if (step==0) {
        r->dxRMS=r->dxRMSInit;
      }
      // Adaptive step size: halve first, then conditionally increase
      // Net 1.2x when energy decreases, 0.5x when increases
      r->dxRMS*=0.5;
      if (step>0 && currEnergy<prevEnergy) {
        r->dxRMS*=2.4;
      }
      /* // RLH: I can't find this condition for stalled minimization in the documentation or steepd.F90
      // Detect stalled minimization (energy unchanged) - apply extra damping
      if (step>0 && currEnergy==prevEnergy) {
        r->dxRMS*=0.5;
        printlog("WARNING: Energy unchanged at step %d, applying extra damping\n", step);
      }
      // Cap dxRMS to prevent unbounded growth (max 10x initial value)
      if (r->dxRMS > 10.0 * r->dxRMSInit) {
        r->dxRMS = 10.0 * r->dxRMSInit;
      } */
      prevEnergy=currEnergy;
      sd_acceleration_kernel<<<(3*atomCount+BLUP-1)/BLUP,BLUP,
        0,r->updateStream>>>(3*atomCount,*leapState);
      gpuCheck(cudaGetLastError());
      holonomic_velocity(system);
      sd_scaling_kernel<<<(atomCount+BLUP-1)/BLUP,BLUP,
        BLUP*sizeof(real)/32,r->updateStream>>>
        (atomCount,*leapState,grads2_d);
      gpuCheck(cudaGetLastError());
      gpuCheck(cudaMemcpy(grads2,grads2_d,2*sizeof(real_e),cudaMemcpyDeviceToHost));
      gpuCheck(cudaMemset(grads2_d,0,2*sizeof(real_e)));
      gradRMS=sqrt(grads2[0]);
      gradMax=sqrt(grads2[1]);

      // scaling factor to achieve desired rms displacement
      scaling=r->dxRMS/gradRMS;
      if (system->verbose > 0) {
        printlog("rmsgrad = %f\n",gradRMS);
        printlog("maxgrad = %f\n",gradMax);
        printlog("scaling = %f\n",scaling);
      }
      sd_position_kernel<<<(3*atomCount+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>
        (3*atomCount,*leapState,leapState->v,scaling,positionCons_d);
      gpuCheck(cudaGetLastError());
      holonomic_position(system);
    }
  } else if (r->minType==esdfd) {
    if (system->id==0) {
      if (step==0) {
        r->dxRMS=r->dxRMSInit;
      }

      sd_acceleration_kernel<<<(3*atomCount+BLUP-1)/BLUP,BLUP,
        0,r->updateStream>>>(3*atomCount,*leapState);
      gpuCheck(cudaGetLastError());
      holonomic_velocity(system);
      gpuCheck(cudaMemcpy(minDirection_d,leapState->v,3*atomCount*sizeof(real_v),cudaMemcpyDeviceToDevice));
      sd_scaling_kernel<<<(atomCount+BLUP-1)/BLUP,BLUP,
        BLUP*sizeof(real)/32,r->updateStream>>>
        (atomCount,*leapState,grads2_d);
      gpuCheck(cudaGetLastError());
      gpuCheck(cudaMemcpy(grads2,grads2_d,2*sizeof(real_e),cudaMemcpyDeviceToHost));
      gpuCheck(cudaMemset(grads2_d,0,2*sizeof(real_e)));
      gradRMS=sqrt(grads2[0]);
      gradMax=sqrt(grads2[1]);

      if (system->verbose > 0) {
        printlog("rmsgrad = %f\n",gradRMS);
        printlog("maxgrad = %f\n",gradMax);
      }
      // scaling factor to achieve desired rms displacement
      scaling=r->dxRMS/gradRMS;
      // ratio of allowed maximum displacement over actual maximum displacement
      rescaling=r->dxAtomMax/(scaling*gradMax);
      if (system->verbose > 0) {
        printlog("scaling = %f, rescaling = %f\n",scaling,rescaling);
      }
      // decrease scaling factor if actual max violates allowed max
      if (rescaling<1) {
        scaling*=rescaling;
        r->dxRMS*=rescaling;
      }
      backup_position();
      sd_position_kernel<<<(3*atomCount+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>
        (3*atomCount,*leapState,leapState->v,scaling,positionCons_d);
      gpuCheck(cudaGetLastError());
      holonomic_position(system);
    }
    system->domdec->update_domdec(system,false); // false, no need to update neighbor list
    system->potential->calc_force(0,system); // step 0 to always calculate energy
    if (system->id==0) {
      sd_acceleration_kernel<<<(3*atomCount+BLUP-1)/BLUP,BLUP,
        0,r->updateStream>>>(3*atomCount,*leapState);
      gpuCheck(cudaGetLastError());
      holonomic_velocity(system);
      sdfd_dotproduct_kernel<<<(3*atomCount+BLUP-1)/BLUP,BLUP,
        BLUP*sizeof(real)/32,r->updateStream>>>
        (3*atomCount,*leapState,minDirection_d,grads2_d);
      gpuCheck(cudaGetLastError());
      gpuCheck(cudaMemcpy(gradDot,grads2_d,sizeof(real_e),cudaMemcpyDeviceToHost));
      gpuCheck(cudaMemset(grads2_d,0,2*sizeof(real_e)));
      // grads2[0] is F*F, gradDot is F*Fnew
      frac=grads2[0]/(grads2[0]-gradDot[0]);
      if (system->verbose > 0) {
        printlog("F(x0)*dx = %f, F(x0+dx)*dx = %f, frac = %f\n",grads2[0],gradDot[0],frac);
      }
      if (frac>1.44 || frac<0) {
        r->dxRMS*=1.2;
      } else if (frac<0.25) {
        r->dxRMS*=0.5;
      } else {
        r->dxRMS*=sqrt(frac);
      }
      frac*=(nsteps-step)/(1.0*nsteps);
      if (frac>1 || frac<0) frac=1;
      restore_position();
      sd_position_kernel<<<(3*atomCount+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>
        (3*atomCount,*leapState,minDirection_d,frac*scaling,positionCons_d);
      gpuCheck(cudaGetLastError());
      holonomic_position(system);
    }
  } else {
    fatal(__FILE__,__LINE__,"Error: Unrecognized minimization type %d\n",r->minType);
  }

  if (system->id==0) {
    // Adaptive MINI> output
    if (step % r->freqNRG == 0) {
      printlog("MINI> Step   Energy         deltaE         grms          \n");
      printlog("MINI> %6d %14.6f %14.6f %14.6f\n", step, currEnergy, deltaE, gradRMS);
    }
  }
}
