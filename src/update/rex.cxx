#include "system/system.h"
#include "system/state.h"
#include "run/run.h"
#include "domdec/domdec.h"
#include "system/potential.h"
#include "rng/rng_cpu.h"

#ifdef REPLICAEXCHANGE
#include <mpi.h>



void replica_exchange(System *system)
{
  int rank, rankCount, rankPartner;
  int direction, localMaster;
  State *s=system->state;
  Run *r=system->run;
  Potential *p=system->potential;
  real_e E[6]; // old master, new master, kT master, old slave, new slave, kT slave
  real dW;
  int n, accept, newReplica;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &rankCount);

  direction=((r->step/r->freqREx)&1);

  localMaster=(direction==(rank&1));
  rankPartner=rank+(2*localMaster-1);

  // Only do the exchange if there is a partner
  if (rankPartner>=0 && rankPartner<rankCount) {
    if (system->id==0) {
      // Get old energy
      s->recv_energy();
      E[0]=s->energy[eepotential];

      // and print it
      /*if (system->verbose>0) {
        for (int i=0; i<eeend; i++) {
          fprintf(stdout," %12.4f",s->energy[i]);
        }
        fprintf(stdout,"\n");
      }*/

      s->backup_position();

      // Swap systems
      // Both positions
      n=2*s->lambdaCount+3*s->atomCount;
      cudaMemcpy(s->positionRExBuffer,s->positionBuffer_d,
        n*sizeof(real_x),cudaMemcpyDeviceToHost);
      MPI_Sendrecv(s->positionRExBuffer,n,MYMPI_REAL_X,rankPartner,10,
        s->positionBuffer,n,MYMPI_REAL_X,rankPartner,10,
        MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      // And boxes
      MPI_Sendrecv((real_x*)&s->boxBackup,6,MYMPI_REAL_X,rankPartner,11,
        (real_x*)&s->box,6,MYMPI_REAL_X,rankPartner,11,
        MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      cudaMemcpy(s->positionBuffer_d,s->positionBuffer,
        n*sizeof(real_x),cudaMemcpyHostToDevice);
    }
    // update_domdec calls broadcast_position
    // Call broadcast_box to set orthBox_f, even if only one node
    s->broadcast_box(system);

    // Evaluate new energy
    system->domdec->update_domdec(system,true);
    system->potential->calc_force(0,system); // 0 tells it to calculate energy freqNRG

    if (system->id==0) {
      // Get new energy
      s->recv_energy();
      E[1]=s->energy[eepotential];

      E[2]=s->leapParms1->kT;
      MPI_Sendrecv(E,3,MYMPI_REAL_E,rankPartner,12,
        E+3,3,MYMPI_REAL_E,rankPartner,12,
        MPI_COMM_WORLD,MPI_STATUS_IGNORE);

      // Compare energy
      dW=(E[1]-E[0])/E[2]+(E[4]-E[3])/E[5];
      if (localMaster) {
        accept=(system->rngCPU->rand_uniform()<exp(-dW));
        MPI_Send(&accept,1,MPI_INT,rankPartner,13,MPI_COMM_WORLD);
      } else {
        MPI_Recv(&accept,1,MPI_INT,rankPartner,13,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      }
      if (accept) { // accept move
        // swap replica indices
        MPI_Sendrecv(&r->replica,1,MPI_INT,rankPartner,14,
          &newReplica,1,MPI_INT,rankPartner,14,
          MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        fprintf(stdout,"Step %d , Rank %d , Replica %d, dW %f, Accept\n",r->step,rank,r->replica,dW);
        r->replica=newReplica;
        // Swap velocities
        n=s->lambdaCount+3*s->atomCount;
        cudaMemcpy(s->positionRExBuffer,s->velocityBuffer_d,
          n*sizeof(real_v),cudaMemcpyDeviceToHost);
        MPI_Sendrecv(s->positionRExBuffer,n,MYMPI_REAL_V,rankPartner,15,
          s->velocityBuffer,n,MYMPI_REAL_V,rankPartner,15,
          MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        cudaMemcpy(s->velocityBuffer_d,s->velocityBuffer,
          n*sizeof(real_v),cudaMemcpyHostToDevice);
      } else {
        // maintain replica indices
        fprintf(stdout,"Step %d , Rank %d , Replica %d, dW %f, Reject\n",r->step,rank,r->replica,dW);
        // Restore positions
        s->restore_position();
      }
    }
    // update_domdec calls broadcast_position
    // Call broadcast_box to set orthBox_f, even if only one node
    system->state->broadcast_box(system);
    // Technically only need to do broadcasts and update domdec if we reject the
    // swap, but that's typically more common anyways. Saves having to
    // communicate between OpenMP threads since only master OpenMP thread knows.
    system->domdec->update_domdec(system,true);
  }

  fprintf(r->fpREx,"%2d\n",r->replica);
}
#endif
