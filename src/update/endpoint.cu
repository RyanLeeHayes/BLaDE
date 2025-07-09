// #include <nvToolsExt.h>

#include "system/system.h"
#include "run/run.h"
#include "system/state.h"
#include "system/potential.h"
#include "domdec/domdec.h"
#include "rng/rng_cpu.h"
#include "holonomic/rectify.h"
#include "io/io.h"
#include "msld/msld.h"



void endpoint_correction(System *system)
{
  State *s=system->state;
  Run *r=system->run;
  Potential *p=system->potential;
  real_e biasOld, biasNew, energyOld, energyNew;

  if (system->id==0) {
    // Get old energy
    s->recv_energy();
    biasOld=s->energy[eelambda];
    energyOld=s->energy[eepotential];

    // display_nrg(system); // DEBUG

    s->backup_position();

    // Set lambdas to enpoint values
    system->msld->calc_endpoint_lambda(r->updateStream,system);
  }
  system->state->broadcast_position(system);

  // Evaluate new energy
  system->domdec->update_domdec(system,false);
  p->calc_force(0,system); // 0 tells it to calculate energy freqNRG

  if (system->id==0) {
    // Get new energy
    s->recv_energy();
    biasNew=s->energy[eelambda];
    energyNew=s->energy[eepotential];

    // display_nrg(system); // DEBUG

    // Compare energy
    fprintf(r->fpEPC,"%12.3f %12.3f %12.3f %12.3f",biasOld,biasNew,energyOld,energyNew);

    // Print endpoint lambdas
    s->recv_lambda();
    int i;
    for (i=1; i<s->lambdaCount; i++) {
      fprintf(r->fpEPC," %8.6f",(real)s->lambda[i]);
    }
    fprintf(r->fpEPC,"\n");

    s->restore_position();
  }
  system->state->broadcast_position(system);
}
