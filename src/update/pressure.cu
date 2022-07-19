// #include <nvToolsExt.h>

#include "system/system.h"
#include "run/run.h"
#include "system/state.h"
#include "system/potential.h"
#include "domdec/domdec.h"
#include "rng/rng_cpu.h"
#include "holonomic/rectify.h"
#include "io/io.h"



real3_x choose_scale_box(System *system,real_x *volumeOut1,real_x *volumeOut2)
{
  real3_x scaleFactor;
  real_x volumeOld, volumeNew;
  if (system->state->typeBox) {
    volumeOld=system->state->tricBox.a.x*system->state->tricBox.b.y*system->state->tricBox.c.z;
  } else {
    volumeOld=system->state->orthBox.x*system->state->orthBox.y*system->state->orthBox.z;
  }
  volumeOut1[0]=volumeOld;
  real volumeFluctuation=system->run->volumeFluctuation;

  int nameBox=system->state->nameBox;
  if (nameBox==ebcubi || nameBox==ebocta || nameBox==ebrhdo) {
    volumeNew=volumeOld+volumeFluctuation*system->rngCPU->rand_normal();
    scaleFactor.x=exp(log(volumeNew/volumeOld)/3);
    scaleFactor.y=scaleFactor.x;
    scaleFactor.z=scaleFactor.x;
  } else if (nameBox==ebtetr || nameBox==ebhexa) {
    volumeNew=volumeOld+volumeFluctuation*system->rngCPU->rand_normal();
    scaleFactor.x=exp(log(volumeNew/volumeOld)/3);
    scaleFactor.y=scaleFactor.x;
    volumeNew=volumeOld+volumeFluctuation*system->rngCPU->rand_normal();
    scaleFactor.z=exp(log(volumeNew/volumeOld)/3);
  } else if (nameBox==eborth) {
    volumeNew=volumeOld+volumeFluctuation*system->rngCPU->rand_normal();
    scaleFactor.x=exp(log(volumeNew/volumeOld)/3);
    volumeNew=volumeOld+volumeFluctuation*system->rngCPU->rand_normal();
    scaleFactor.y=exp(log(volumeNew/volumeOld)/3);
    volumeNew=volumeOld+volumeFluctuation*system->rngCPU->rand_normal();
    scaleFactor.z=exp(log(volumeNew/volumeOld)/3);
  } else if (nameBox==ebmono || nameBox==ebtric || nameBox==ebrhom) {
    fatal(__FILE__,__LINE__,"Pressure coupling does not support angle skewing for monoclinic, triclinic, and rhombohedral box types\n");
  } else {
    fatal(__FILE__,__LINE__,"How did this box type even happen?\n");
  }
  volumeNew=volumeOld*scaleFactor.x*scaleFactor.y*scaleFactor.z;
  volumeOut2[0]=volumeNew;
  return scaleFactor;
}

// scale_box_kernel<<<(N+BLUP-1)/BLUP,BLUP,0,system->run->updateStream>>>(N,scaleFactor,system->state->position_d);
__global__ void scale_box_kernel(int N,real3_x scaleFactor,real3_x *position)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<N) {
    position[i].x*=scaleFactor.x;
    position[i].y*=scaleFactor.y;
    position[i].z*=scaleFactor.z;
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

void scale_box(System *system,real3_x scaleFactor)
{
  system->state->box.a.x*=scaleFactor.x;
  system->state->box.a.y*=scaleFactor.y;
  system->state->box.a.z*=scaleFactor.z;
  // skew these for holonomic_rectify before broadcast_box call
  if (system->state->typeBox) {
    system->state->tricBox.a.x*=scaleFactor.x;
    system->state->tricBox.b.x*=scaleFactor.y;
    system->state->tricBox.b.y*=scaleFactor.y;
    system->state->tricBox.c.x*=scaleFactor.z;
    system->state->tricBox.c.y*=scaleFactor.z;
    system->state->tricBox.c.z*=scaleFactor.z;
  } else {
    system->state->orthBox.x*=scaleFactor.x;
    system->state->orthBox.y*=scaleFactor.y;
    system->state->orthBox.z*=scaleFactor.z;
  }

  int N=system->state->atomCount;
  scale_box_kernel<<<(N+BLUP-1)/BLUP,BLUP,0,system->run->updateStream>>>(N,scaleFactor,(real3_x*)system->state->position_d);

  // Nudge the system to remain centered on absolute harmonic restraints
  if (system->potential->harmCount) {
    real3_x shift;
    shift.x=(1-scaleFactor.x)*system->potential->harmCenter.x;
    shift.y=(1-scaleFactor.y)*system->potential->harmCenter.y;
    shift.z=(1-scaleFactor.z)*system->potential->harmCenter.z;
    shift_box_kernel<<<(N+BLUP-1)/BLUP,BLUP,0,system->run->updateStream>>>(N,shift,(real3_x*)system->state->position_d);
  }

  // There might be better ways to rectify holonomic constraints after volume update, I just want to avoid having bonds change direction, which will mess with the velocities"
  holonomic_rectifyback(system);
}

void pressure_coupling(System *system)
{
  State *s=system->state;
  Run *r=system->run;
  Potential *p=system->potential;
  real_e energyOld, energyNew;
  real_x volumeOld, volumeNew;
  real3_x scaleFactor;
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
    scaleFactor=choose_scale_box(system,&volumeOld,&volumeNew);
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
    dW=energyNew-energyOld+r->pressure*(volumeNew-volumeOld)-N*kT*log(volumeNew/volumeOld);
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
