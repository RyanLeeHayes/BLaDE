// #include <nvToolsExt.h>

#include "system/system.h"
#include "run/run.h"
#include "system/state.h"
#include "system/potential.h"
#include "domdec/domdec.h"
#include "rng/rng_cpu.h"
#include "holonomic/rectify.h"



// scale_box_kernel<<<(N+BLUP-1)/BLUP,BLUP,0,system->run->updateStream>>>(N,scaleFactor,system->state->position_d);
__global__ void scale_box_kernel(int N,real_x scaleFactor,real_x *position)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<N) {
    position[i]*=scaleFactor;
  }
}

__global__ void shift_box_kernel(int N,real3_x shift,real3_x *position)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<N) {
    position[i].x+=shift.x;
    position[i].y+=shift.y;
    position[i].z+=shift.z;
  }
}

void scale_box(System *system,real_x scaleFactor)
{
  system->state->orthBox.x*=scaleFactor;
  system->state->orthBox.y*=scaleFactor;
  system->state->orthBox.z*=scaleFactor;

  int N=3*system->state->atomCount;
  scale_box_kernel<<<(N+BLUP-1)/BLUP,BLUP,0,system->run->updateStream>>>(N,scaleFactor,(real_x*)system->state->position_d);

  // Nudge the system to remain centered on absolute harmonic restraints
  if (system->potential->harmCount) {
    int N3=system->state->atomCount;
    real3_x shift;
    shift.x=(1-scaleFactor)*system->potential->harmCenter.x;
    shift.y=(1-scaleFactor)*system->potential->harmCenter.y;
    shift.z=(1-scaleFactor)*system->potential->harmCenter.z;
    shift_box_kernel<<<(N3+BLUP-1)/BLUP,BLUP,0,system->run->updateStream>>>(N3,shift,(real3_x*)system->state->position_d);
  }

  // There might be better ways to rectify holonomic constraints after volume update, I just want to avoid having bonds change direction, which will mess with the velocities"
  holonomic_rectify(system);
}

void pressure_coupling(System *system)
{
  State *s=system->state;
  Run *r=system->run;
  Potential *p=system->potential;
  real_e energyOld, energyNew;
  real_x volumeOld, volumeNew;
  real_x scaleFactor;
  real N,kT,dW;

  // cudaStreamWaitEvent(r->updateStream,r->forceComplete,0);

  // nvtxRangePushA("pressure_coupling");
  if (system->id==0) {
    // nvtxRangePushA("head node stuff...");
    // Get old energy
    s->recv_energy();
    energyOld=s->energy[eepotential];

    // and print it
    if (system->verbose>0) {
      for (int i=0; i<eeend; i++) {
        fprintf(stdout," %12.4f",s->energy[i]);
      }
      fprintf(stdout,"\n");
    }

    s->backup_position();

    // Change volume
    volumeOld=s->orthBox.x*s->orthBox.y*s->orthBox.z;
    volumeNew=volumeOld+r->volumeFluctuation*system->rngCPU->rand_normal();
    scaleFactor=exp(log(volumeNew/volumeOld)/3);
    scale_box(system,scaleFactor);
    // nvtxRangePop();
  }
  // Call broadcast_box to set orthBox_f, even if only one node
  system->state->broadcast_box(system);

  // Evaluate new energy
  // nvtxRangePushA("update_domdec");
  system->domdec->update_domdec(system,false);
  // nvtxRangePop();
  // nvtxRangePushA("calc_force");
  p->calc_force(0,system); // 0 tells it to calculate energy freqNRG
  // nvtxRangePop();

  if (system->id==0) {
    // nvtxRangePushA("more head node stuff...");
    // Get new energy
    s->recv_energy();
    energyNew=s->energy[eepotential];

    // and print it
    if (system->verbose>0) {
      for (int i=0; i<eeend; i++) {
        fprintf(stdout," %12.4f",s->energy[i]);
      }
      fprintf(stdout,"\n");
    }

    // Compare energy
    N=s->atomCount-(2*p->triangleConsCount+p->branch1ConsCount+2*p->branch2ConsCount+3*p->branch3ConsCount);
    kT=s->leapParms1->kT;
    dW=energyNew-energyOld+system->run->pressure*(volumeNew-volumeOld)-N*kT*log(volumeNew/volumeOld);
    if (system->verbose>0) {
      fprintf(stdout,"dW= %f, dV= %f\n",dW,volumeNew-volumeOld);
    }
    if (system->rngCPU->rand_uniform()<exp(-dW/kT)) { // accept move
      if (system->verbose>0) {
        fprintf(stdout,"Volume move accepted. New volume=%f\n",volumeNew);
      }
    } else {
      if (system->verbose>0) {
        fprintf(stdout,"Volume move rejected. Old volume=%f\n",volumeOld);
      }
      s->restore_position();
    }
    // nvtxRangePop();
  }
  // Call broadcast_box to set orthBox_f, even if only one node
  system->state->broadcast_box(system);
  // nvtxRangePop();
}
