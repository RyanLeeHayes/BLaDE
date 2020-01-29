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
  real E[6]; // old master, new master, kT master, old slave, new slave, kT slave
  real dW;
  int accept, newReplica;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &rankCount);

  direction=((r->step/r->freqREx)&1);

  localMaster=(direction==(rank&1));
  rankPartner=rank+(2*localMaster-1);

  // Only do the exchange if there is a partner
  if (rankPartner>=0 && rankPartner<rankCount) {
    int n=2*s->lambdaCount+3*s->atomCount;
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
      cudaMemcpy(s->positionRExBuffer,s->positionBuffer_d,
        n*sizeof(real),cudaMemcpyDeviceToHost);
      MPI_Sendrecv(s->positionRExBuffer,n,MYMPI_REAL,rankPartner,10,
        s->positionBuffer,n,MYMPI_REAL,rankPartner,10,
        MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      // And boxes
      MPI_Sendrecv((real*)&s->orthBoxBackup,3,MYMPI_REAL,rankPartner,11,
        (real*)&s->orthBox,3,MYMPI_REAL,rankPartner,11,
        MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      cudaMemcpy(s->positionBuffer_d,s->positionRExBuffer,
        n*sizeof(real),cudaMemcpyHostToDevice);
    }
    if (system->idCount>1) {
      s->broadcast_position(system);
      s->broadcast_box(system);
    }

    // Evaluate new energy
    system->domdec->update_domdec(system,true);
    system->potential->calc_force(0,system); // 0 tells it to calculate energy freqNRG

    if (system->id==0) {
      // Get new energy
      s->recv_energy();
      E[1]=s->energy[eepotential];

      // and print it
      /*if (system->verbose>0) {
        for (int i=0; i<eeend; i++) {
          fprintf(stdout," %12.4f",s->energy[i]);
        }
        fprintf(stdout,"\n");
      }*/

      E[2]=s->leapParms1->kT;
      MPI_Sendrecv(E,3,MYMPI_REAL,rankPartner,12,
        E+3,3,MYMPI_REAL,rankPartner,12,
        MPI_COMM_WORLD,MPI_STATUS_IGNORE);

      // Compare energy
      dW=(E[1]-E[0])/E[2]+(E[4]-E[3])/E[5];
      /*if (system->verbose>0) {
        fprintf(stdout,"dW= %f, (Boltzman Factor Exponent for Exchange)\n",dW);
      }*/
      if (localMaster) {
        accept=(system->rngCPU->rand_uniform()<exp(-dW));
        MPI_Send(&accept,1,MPI_INT,rankPartner,13,MPI_COMM_WORLD);
      } else {
        MPI_Recv(&accept,1,MPI_INT,rankPartner,13,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      }
      if (accept) { // accept move
        MPI_Sendrecv(&r->replica,1,MPI_INT,rankPartner,14,
          &newReplica,1,MPI_INT,rankPartner,14,
          MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        /*if (system->verbose>0) {
          fprintf(stdout,"Volume move accepted. New volume=%f\n",volumeNew);
        }*/
        fprintf(stdout,"Step %d , Rank %d , Replica %d, dW %f, Accept, E %f %f %f %f %f %f\n",r->step,rank,r->replica,dW,E[0],E[1],E[2],E[3],E[4],E[5]);
        r->replica=newReplica;
      } else {
        /*if (system->verbose>0) {
          fprintf(stdout,"Volume move rejected. Old volume=%f\n",volumeOld);
        }*/
        fprintf(stdout,"Step %d , Rank %d , Replica %d, dW %f, Reject, E %f %f %f %f %f %f\n",r->step,rank,r->replica,dW,E[0],E[1],E[2],E[3],E[4],E[5]);
        s->restore_position();
      }
    }
    if (system->idCount>1) {
      system->state->broadcast_box(system);
    }
  }

  fprintf(r->fpREx,"%2d\n",r->replica);
}
#endif
