#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

#include "system/system.h"
#include "system/state.h"
#include "run/run.h"
#include "system/potential.h"
#include "holonomic/holonomic.h"
#include "io/io.h"

#include "main/real3.h"



__global__ void sd_acceleration_kernel(int N,struct LeapState ls,real_e *displacements2)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;

  if (i<N) {
    if (isfinite(ls.ism[i])) {
      ls.v[i]=-ls.f[i]*ls.ism[i]*ls.ism[i];
    } else {
      ls.v[i]=0; // No acceleration for massless virtual particles
    }
  }
}

__global__ void sd_scaling_kernel(int N,struct LeapState ls,real_e *displacements2)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real3 displacement;
  real displacement2=0;
  extern __shared__ real sDisplacement2[];

  if (i<N) {
    displacement.x=ls.v[3*i+0];
    displacement.y=ls.v[3*i+1];
    displacement.z=ls.v[3*i+2];
    displacement2 =displacement.x*displacement.x;
    displacement2+=displacement.y*displacement.y;
    displacement2+=displacement.z*displacement.z;
  }

  real_sum_reduce(displacement2/N,sDisplacement2,displacements2);
  real_max_reduce(displacement2,sDisplacement2,displacements2+1);
}

__global__ void sd_position_kernel(int N,struct LeapState ls,real scale,real_x *bx)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real_x x;

  if (i<N) {
    x=ls.x[i];
    if (bx) bx[i]=x;
    ls.x[i]=x+scale*ls.v[i];
  }
}

void State::min_init(System *system)
{
  if (system->run->minType==esd) {
    cudaMalloc(&displacements2_d,2*sizeof(real_e));
    cudaMemset(displacements2_d,0,2*sizeof(real_e));
  }
}

void State::min_dest(System *system)
{
  if (system->run->minType==esd) {
    cudaFree(displacements2_d);
  }
}

void State::min_move(int step,System *system)
{
  Run *r=system->run;
  real_e displacements2[2];
  real_e currEnergy;
  real scaling, rescaling;

  if (r->minType==esd) {
    if (system->id==0) {
      recv_energy();
      if (system->verbose>0) {
        for (int i=0; i<eeend; i++) {
          fprintf(stdout," %12.4f",energy[i]);
        }
        fprintf(stdout,"\n");
      }
      currEnergy=energy[eepotential];
      if (step==0) {
        r->dxRMS=r->dxRMSInit;
      } else if (currEnergy<prevEnergy) {
        r->dxRMS*=1.2;
      } else {
        r->dxRMS*=0.5;
      }
      prevEnergy=currEnergy;
      sd_acceleration_kernel<<<(3*atomCount+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(atomCount,*leapState,displacements2_d);
      holonomic_velocity(system);
      sd_scaling_kernel<<<(3*atomCount+BLUP-1)/BLUP,BLUP,BLUP*sizeof(real)/32,r->updateStream>>>(atomCount,*leapState,displacements2_d);
      cudaMemcpy(displacements2,displacements2_d,2*sizeof(real_e),cudaMemcpyDeviceToHost);
      cudaMemset(displacements2_d,0,2*sizeof(real_e));
      fprintf(stdout,"rmsdisplacement = %f\n",sqrt(displacements2[0]));
      fprintf(stdout,"maxdisplacement = %f\n",sqrt(displacements2[1]));
      // scaling factor to achieve desired rms displacement
      scaling=r->dxRMS/sqrt(displacements2[0]);
      // ratio of allowed maximum displacement over actual maximum displacement
      rescaling=r->dxAtomMax/(scaling*sqrt(displacements2[1]));
      fprintf(stdout,"scaling = %f, rescaling = %f\n",scaling,rescaling);
      // decrease scaling factor if actual max violates allowed max
      if (rescaling<1) {
        scaling*=rescaling;
        r->dxRMS*=rescaling;
        // r->dxRMS*=(0.5*rescaling); // try a shorter step next time
      }
      sd_position_kernel<<<(3*atomCount+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(3*atomCount,*leapState,scaling,positionCons_d);
      holonomic_position(system);
    }
  } else {
    fatal(__FILE__,__LINE__,"Error: Unrecognized minimization type %d\n",r->minType);
  }
}
