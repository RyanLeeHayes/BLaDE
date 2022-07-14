// #include <nvToolsExt.h>
#include <string.h>
#include <math.h>

#include "system/state.h"
#include "io/io.h"
#include "system/system.h"
#include "system/structure.h"
#include "msld/msld.h"
#include "system/coordinates.h"
#include "run/run.h"
#include "system/potential.h"
#include "rng/rng_cpu.h"
#include "rng/rng_gpu.h"



// Class constructors
State::State(System *system) {
  int i;

  atomCount=system->coordinates->atomCount;
  lambdaCount=system->msld->blockCount;

  int n=atomCount;
  int nL=lambdaCount;
  int rootFactor=(system->id==0?system->idCount:1);

  // Buffers for transfer and reduction

  // Lambda-Spatial-Theta buffers
  positionBuffer=(real_x*)calloc((2*nL+3*n),sizeof(real_x));
  cudaMalloc(&(positionBuffer_d),(2*nL+3*n)*sizeof(real_x));
  if (sizeof(real)==sizeof(real_x)) {
    positionBuffer_fd=(real*)positionBuffer_d;
  } else {
    cudaMalloc(&(positionBuffer_fd),(2*nL+3*n)*sizeof(real));
  }
#ifdef REPLICAEXCHANGE
  positionRExBuffer=(real_x*)calloc((2*nL+3*n),sizeof(real_x));
#endif
  cudaMalloc(&(positionBackup_d),(2*nL+3*n)*sizeof(real_x));
  forceBuffer=(real_f*)calloc(rootFactor*(2*nL+3*n),sizeof(real_f));
  cudaMalloc(&(forceBuffer_d),rootFactor*(2*nL+3*n)*sizeof(real_f));
  cudaMalloc(&(forceBackup_d),(2*nL+3*n)*sizeof(real_f));

  if (system->idCount>0) { // OMP
#pragma omp barrier // OMP
    if (system->id==0) { // OMP
      system->message[0]=(void*)positionBuffer_fd; // OMP
      for (i=1; i<system->idCount; i++) { // OMP
        system->message[i]=(void*)&forceBuffer_d[i*(2*nL+3*n)]; // OMP
      } // OMP
    }
#pragma omp barrier
    if (system->id!=0) { // OMP
      positionBuffer_omp=(real*)(system->message[0]); // OMP
      forceBuffer_omp=(real_f*)(system->message[system->id]); // OMP
    } // OMP
#pragma omp barrier // OMP
  } // OMP

  // Other buffers
  energy=(real_e*)calloc(rootFactor*eeend,sizeof(real_e));
  cudaMalloc(&(energy_d),rootFactor*eeend*sizeof(real_e));
  cudaMalloc(&(energyBackup_d),eeend*sizeof(real_e));

  if (system->idCount>0) { // OMP
#pragma omp barrier // OMP
    if (system->id==0) { // OMP
      system->message[0]=(void*)&orthBox_f; // OMP
      for (i=1; i<system->idCount; i++) { // OMP
        system->message[i]=(void*)&energy_d[i*eeend]; // OMP
      } // OMP
    }
#pragma omp barrier
    if (system->id!=0) { // OMP
      orthBox_omp=(real3*)(system->message[0]); // OMP
      energy_omp=(real_e*)(system->message[system->id]); // OMP
    } // OMP
#pragma omp barrier // OMP
    if (system->id==0) { // OMP
      system->message[0]=(void*)&tricBox_f; // OMP
    }
#pragma omp barrier
    if (system->id!=0) { // OMP
      tricBox_omp=(real123*)(system->message[0]); // OMP
    } // OMP
#pragma omp barrier // OMP
  } // OMP

  // Spatial-Theta buffers
  velocityBuffer=(real_v*)calloc((nL+3*n),sizeof(real_v));
  cudaMalloc(&(velocityBuffer_d),(nL+3*n)*sizeof(real_v));
  invsqrtMassBuffer=(real*)calloc((nL+3*n),sizeof(real));
  cudaMalloc(&(invsqrtMassBuffer_d),(nL+3*n)*sizeof(real));

  // Constraint stuff
  positionCons_d=NULL;
  if (system->structure->shakeHbond) {
    cudaMalloc(&(positionCons_d),(nL+3*n)*sizeof(real_x));
  }

  // The box
  nameBox=system->coordinates->particleBoxName;
  box.a=system->coordinates->particleBoxABC;
  box.b=system->coordinates->particleBoxAlBeGa;
  typeBox=(nameBox!=ebcubi && nameBox!=ebtetr && nameBox!=eborth);
  check_box(system);

  // Buffer for floating point output
  positionXTC=(float(*)[3])calloc(n,sizeof(float[3])); // intentional float

  // Labels (do not free)
  lambda=positionBuffer;
  lambda_d=positionBuffer_d;
  lambda_fd=positionBuffer_fd;
  position=(real_x(*)[3])(positionBuffer+nL);
  position_d=(real_x(*)[3])(positionBuffer_d+nL);
  position_fd=(real(*)[3])(positionBuffer_fd+nL);
  positionb_d=(real_x(*)[3])(positionBackup_d+nL);
  theta=positionBuffer+nL+3*n;
  theta_d=positionBuffer_d+nL+3*n;
  theta_fd=positionBuffer_fd+nL+3*n;
  velocity=(real_v(*)[3])velocityBuffer;
  velocity_d=(real_v(*)[3])velocityBuffer_d;
  thetaVelocity=velocityBuffer+3*n;
  thetaVelocity_d=velocityBuffer_d+3*n;
  lambdaForce=forceBuffer;
  lambdaForce_d=forceBuffer_d;
  force=(real_f(*)[3])(forceBuffer+nL);
  force_d=(real_f(*)[3])(forceBuffer_d+nL);
  thetaForce=forceBuffer+nL+3*n;
  thetaForce_d=forceBuffer_d+nL+3*n;
  invsqrtMass=(real(*)[3])invsqrtMassBuffer;
  invsqrtMass_d=(real(*)[3])invsqrtMassBuffer_d;
  thetaInvsqrtMass=invsqrtMassBuffer+3*n;
  thetaInvsqrtMass_d=invsqrtMassBuffer_d+3*n;

  // Leapfrog structures
  leapParms1=alloc_leapparms1(system->run->dt,system->run->gamma,system->run->T);
  leapParms2=alloc_leapparms2(system->run->dt,system->run->gamma,system->run->T);
  lambdaLeapParms1=alloc_leapparms1(system->run->dt,system->msld->gamma,system->run->T);
  lambdaLeapParms2=alloc_leapparms2(system->run->dt,system->msld->gamma,system->run->T);
  leapState=alloc_leapstate(3*atomCount,lambdaCount,(real_x*)position_d,velocityBuffer_d,(real_f*)force_d,invsqrtMassBuffer_d);
}

State::~State() {
  // Lambda-Spatial-Theta buffers
  if (positionBuffer) free(positionBuffer);
  if (positionBuffer_d) cudaFree(positionBuffer_d);
  if ((void*)positionBuffer_fd!=(void*)positionBuffer_d) cudaFree(positionBuffer_fd);
#ifdef REPLICAEXCHANGE
  if (positionRExBuffer) free(positionRExBuffer);
#endif
  if (positionBackup_d) cudaFree(positionBackup_d);
  if (forceBuffer) free(forceBuffer);
  if (forceBuffer_d) cudaFree(forceBuffer_d);
  if (forceBackup_d) cudaFree(forceBackup_d);
  // Other buffers
  if (energy) free(energy);
  if (energy_d) cudaFree(energy_d);
  if (energyBackup_d) cudaFree(energyBackup_d);
  // Spatial-Theta buffers
  if (velocityBuffer) free(velocityBuffer);
  if (velocityBuffer_d) cudaFree(velocityBuffer_d);
  if (invsqrtMassBuffer) free(invsqrtMassBuffer);
  if (invsqrtMassBuffer_d) cudaFree(invsqrtMassBuffer_d);
  // Constraint stuff
  if (positionCons_d) cudaFree(positionCons_d);
  // Buffer for floating point output
  if (positionXTC) free(positionXTC);

  if (leapParms1) free(leapParms1);
  if (leapParms2) free(leapParms2);
  if (lambdaLeapParms1) free(lambdaLeapParms1);
  if (lambdaLeapParms2) free(lambdaLeapParms2);
  if (leapState) free_leapstate(leapState);
}

void State::initialize(System *system)
{
  int i,j;
  int n=atomCount;
  int nL=lambdaCount;

  for (i=0; i<atomCount; i++) {
    for (j=0; j<3; j++) {
      position[i][j]=system->coordinates->particlePosition[i][j];
      velocity[i][j]=system->coordinates->particleVelocity[i][j];
      invsqrtMass[i][j]=1/sqrt(system->structure->atomList[i].mass);
    }
  }
  for (i=0; i<lambdaCount; i++) {
    theta[i]=system->msld->theta[i];
    thetaVelocity[i]=system->msld->thetaVelocity[i];
    thetaInvsqrtMass[i]=1/sqrt(system->msld->thetaMass[i]);
  }

  cudaMemcpy(positionBuffer_d,positionBuffer,(2*nL+3*n)*sizeof(real_x),cudaMemcpyHostToDevice);
  cudaMemcpy(velocityBuffer_d,velocityBuffer,(nL+3*n)*sizeof(real_v),cudaMemcpyHostToDevice);
  cudaMemset(forceBuffer_d,0,(nL+3*n)*sizeof(real_f));
  cudaMemcpy(invsqrtMassBuffer_d,invsqrtMassBuffer,(nL+3*n)*sizeof(real),cudaMemcpyHostToDevice);

  if (system->msld->fix) { // ffix
    cudaMemcpy(lambda_d,theta,nL*sizeof(real_x),cudaMemcpyHostToDevice);
  }
#warning "Running nvprof on 2080s causes seg faults in the next command and at later locations"
  system->msld->calc_lambda_from_theta(0,system);
}

void State::save_state(System *system)
{
  int i,j;
  int n=atomCount;
  int nL=lambdaCount;

  cudaMemcpy(positionBuffer,positionBuffer_d,(2*nL+3*n)*sizeof(real_x),cudaMemcpyDeviceToHost);
  cudaMemcpy(velocityBuffer,velocityBuffer_d,(nL+3*n)*sizeof(real_v),cudaMemcpyDeviceToHost);

  for (i=0; i<atomCount; i++) {
    for (j=0; j<3; j++) {
       system->coordinates->particlePosition[i][j]=position[i][j];
       system->coordinates->particleVelocity[i][j]=velocity[i][j];
    }
  }
  system->coordinates->particleBoxABC=box.a;
  system->coordinates->particleBoxAlBeGa=box.b;
  for (i=0; i<lambdaCount; i++) {
    system->msld->theta[i]=theta[i];
    system->msld->thetaVelocity[i]=thetaVelocity[i];
  }
}



struct LeapParms1* State::alloc_leapparms1(real dt,real gamma,real T)
{
  struct LeapParms1 *lp;
  real kT=kB*T;

  lp=(struct LeapParms1*) malloc(sizeof(struct LeapParms1));

  lp->dt=dt;
  lp->gamma=gamma;
  lp->kT=kT;

  return lp;
}

struct LeapParms2* State::alloc_leapparms2(real dt,real gamma,real T)
{
  struct LeapParms2 *lp;
  real kT=kB*T;
  // Original integrator from https://pubs.acs.org/doi/10.1021/jp411770f
  // Current integrator from http://dx.doi.org/10.1098/rspa.2016.0138
  real a2=exp(-gamma*dt);

  lp=(struct LeapParms2*) malloc(sizeof(struct LeapParms2));

  lp->friction=a2;
  lp->noise=sqrt((1-a2*a2)*kT);
  lp->halfdt=0.5*dt;

  return lp;
}

struct LeapState* State::alloc_leapstate(int N1,int N2,real_x *x,real_v *v,real_f *f,real *ism)
{
  struct LeapState *ls;

  ls=(struct LeapState*) malloc(sizeof(struct LeapState));
  real *random;
  cudaMalloc(&random,(N1+N2)*sizeof(real));

  ls->N1=N1;
  ls->N=N1+N2;
  ls->x=x;
  ls->v=v;
  ls->f=f;
  ls->ism=ism;
  ls->random=random;
  return ls;
}

void State::free_leapstate(struct LeapState* ls)
{
  cudaFree(ls->random);
}

void State::recv_state()
{
  cudaMemcpy(theta,theta_d,lambdaCount*sizeof(real_x),cudaMemcpyDeviceToHost);
  cudaMemcpy(position,position_d,3*atomCount*sizeof(real_x),cudaMemcpyDeviceToHost);
  cudaMemcpy(thetaVelocity,thetaVelocity_d,lambdaCount*sizeof(real_v),cudaMemcpyDeviceToHost);
  cudaMemcpy(velocity,velocity_d,3*atomCount*sizeof(real_v),cudaMemcpyDeviceToHost);
}

void State::send_state()
{
  cudaMemcpy(theta_d,theta,lambdaCount*sizeof(real_x),cudaMemcpyHostToDevice);
  cudaMemcpy(position_d,position,3*atomCount*sizeof(real_x),cudaMemcpyHostToDevice);
  cudaMemcpy(thetaVelocity_d,thetaVelocity,lambdaCount*sizeof(real_v),cudaMemcpyHostToDevice);
  cudaMemcpy(velocity_d,velocity,3*atomCount*sizeof(real_v),cudaMemcpyHostToDevice);
}

void State::recv_position()
{
  cudaMemcpy(position,position_d,3*atomCount*sizeof(real_x),cudaMemcpyDeviceToHost);
}

void State::recv_lambda()
{
  cudaMemcpy(lambda,lambda_d,lambdaCount*sizeof(real_x),cudaMemcpyDeviceToHost);
}

void State::recv_energy()
{
  int i;

  cudaMemcpy(energy,energy_d,eeend*sizeof(real_e),cudaMemcpyDeviceToHost);

  for (i=0; i<eepotential; i++) {
    energy[eepotential]+=energy[i];
  }
  energy[eetotal]=energy[eepotential]+energy[eekinetic];
}


void State::backup_position()
{
  cudaMemcpy(positionBackup_d,positionBuffer_d,(2*lambdaCount+3*atomCount)*sizeof(real_x),cudaMemcpyDeviceToDevice);
  cudaMemcpy(forceBackup_d,forceBuffer_d,(2*lambdaCount+3*atomCount)*sizeof(real_f),cudaMemcpyDeviceToDevice);
  cudaMemcpy(energyBackup_d,energy_d,eeend*sizeof(real_e),cudaMemcpyDeviceToDevice);
  boxBackup=box;
  if (typeBox) {
    tricBoxBackup=tricBox;
  } else {
    orthBoxBackup=orthBox;
  }
}

void State::restore_position()
{
  cudaMemcpy(positionBuffer_d,positionBackup_d,(2*lambdaCount+3*atomCount)*sizeof(real_x),cudaMemcpyDeviceToDevice);
  cudaMemcpy(forceBuffer_d,forceBackup_d,(2*lambdaCount+3*atomCount)*sizeof(real_f),cudaMemcpyDeviceToDevice);
  cudaMemcpy(energy_d,energyBackup_d,eeend*sizeof(real_e),cudaMemcpyDeviceToDevice);
  box=boxBackup;
  if (typeBox) {
    tricBox=tricBoxBackup;
  } else {
    orthBox=orthBoxBackup;
  }
}

// CUDA AWARE MPI // NOMPI
// https://www.open-mpi.org/faq/?category=runcuda#mpi-cuda-aware-support // NOMPI
// doesn't work well

void State::broadcast_position(System *system)
{
  int N=3*atomCount+2*lambdaCount;
  // nvtxRangePushA("broadcast_position");
/*  if (system->id==0) {
    cudaMemcpy(positionBuffer,positionBuffer_d,N*sizeof(real),cudaMemcpyDeviceToHost);
  }
  MPI_Bcast(positionBuffer,N,MPI_CREAL,0,MPI_COMM_WORLD); // NOMPI
  if (system->id!=0) {
    cudaMemcpy(positionBuffer_d,positionBuffer,N*sizeof(real),cudaMemcpyHostToDevice);
  }*/ // NOMPI
  if (system->id==0) {
    set_fd(system);
    cudaEventRecord(system->run->communicate,system->run->updateStream);
  }
#pragma omp barrier
  if (system->id!=0) { // OMP
#define ORIGINAL_BROADCAST
#ifdef ORIGINAL_BROADCAST
    cudaEventSynchronize(system->run->communicate_omp[0]);
    // cudaMemcpyPeer(positionBuffer_d,system->id,positionBuffer_omp,0,N*sizeof(real)); // OMP
    cudaMemcpyAsync(positionBuffer_fd,positionBuffer_omp,N*sizeof(real),cudaMemcpyDefault,system->run->updateStream); // OMP
#else
    cudaStreamWaitEvent(system->run->updateStream,system->run->communicate_omp[0],0); // UNSURE
    // cudaMemcpyPeer(positionBuffer_d,system->id,positionBuffer_omp,0,N*sizeof(real)); // OMP
    // cudaMemcpyAsync(positionBuffer_fd,positionBuffer_omp,N*sizeof(real),cudaMemcpyDefault,system->run->updateStream); // OMP
    cudaMemcpyPeerAsync(positionBuffer_fd,system->id,positionBuffer_omp,0,N*sizeof(real),system->run->updateStream); // OMP
#endif
  } // OMP
  // nvtxRangePop();
}

void State::broadcast_box(System *system)
{
  // int N=3; // NOMPI
  // MPI_Bcast(&orthBox,N,MPI_CREAL,0,MPI_COMM_WORLD); // NOMPI
  // nvtxRangePushA("broadcast_box");
  if (system->id==0) {
    if (typeBox) {
      tricBox.a.x=box.a.x;
      tricBox.b.x=box.a.y*cos(box.b.z*DEGREES);
      tricBox.b.y=box.a.y*sin(box.b.z*DEGREES);
      tricBox.c.x=box.a.z*cos(box.b.y*DEGREES);
      tricBox.c.y=box.a.z*(cos(box.b.x*DEGREES)-cos(box.b.y*DEGREES)*cos(box.b.z*DEGREES))/sin(box.b.z*DEGREES);
      tricBox.c.z=sqrt(box.a.z*box.a.z-tricBox.c.x*tricBox.c.x-tricBox.c.y*tricBox.c.y);

      // Ensure lattice vectors are in minimal representation
      // Hardcode so rounding errors don't mess some of the represenations up
      if (nameBox==ebrhdo) {
        // This happens to be the 90 60 60 choice for rhdo
        tricBox.c.x+=-tricBox.b.x;
        tricBox.c.y+=-tricBox.b.y;
        tricBox.c.x+=tricBox.a.x;
      } else if (nameBox==ebocta || nameBox==ebhexa) {
        ; // Do nothing, they're fine
      } else {
        int im;
        im=floor(tricBox.c.y/tricBox.b.y+0.5);
        tricBox.c.x+=-im*tricBox.b.x;
        tricBox.c.y+=-im*tricBox.b.y;
        im=floor(tricBox.c.x/tricBox.a.x+0.5);
        tricBox.c.x+=-im*tricBox.a.x;
        im=floor(tricBox.b.x/tricBox.a.x+0.5);
        tricBox.b.x+=-im*tricBox.a.x;
      }

      tricBox_f.a.x=tricBox.a.x;
      tricBox_f.b.x=tricBox.b.x;
      tricBox_f.b.y=tricBox.b.y;
      tricBox_f.c.x=tricBox.c.x;
      tricBox_f.c.y=tricBox.c.y;
      tricBox_f.c.z=tricBox.c.z;

      real Vinv=1/(tricBox_f.a.x*tricBox_f.b.y*tricBox_f.c.z);
      kTricBox_f.a.x=tricBox_f.b.y*tricBox_f.c.z*Vinv;
      kTricBox_f.a.y=-tricBox_f.b.x*tricBox_f.c.z*Vinv;
      kTricBox_f.a.z=(tricBox_f.b.x*tricBox_f.c.y-tricBox_f.b.y*tricBox_f.c.x)*Vinv;
      kTricBox_f.b.x=tricBox_f.c.z*tricBox_f.a.x*Vinv; // actually b.y
      kTricBox_f.b.y=-tricBox_f.c.y*tricBox_f.a.x*Vinv; // actually b.z
      kTricBox_f.c.x=tricBox_f.a.x*tricBox_f.b.y*Vinv; // actually c.z
    } else {
      orthBox=box.a;

      orthBox_f.x=orthBox.x;
      orthBox_f.y=orthBox.y;
      orthBox_f.z=orthBox.z;

      kOrthBox_f.x=1/orthBox_f.x;
      kOrthBox_f.y=1/orthBox_f.y;
      kOrthBox_f.z=1/orthBox_f.z;
    }
  }
#pragma omp barrier // OMP
  if (system->id!=0) { // OMP
    if (typeBox) {
      orthBox_f=orthBox_omp[0]; // OMP
    } else {
      tricBox_f=tricBox_omp[0]; // OMP
    }
  } // OMP
  // nvtxRangePop();
}

void State::gather_force(System *system,bool calcEnergy)
{
  int N=3*atomCount+2*system->msld->blockCount;
  // nvtxRangePushA("gather_force");
/*  if (system->id!=0) {
    cudaMemcpy(forceBuffer,forceBuffer_d,N*sizeof(real),cudaMemcpyDeviceToHost);
    if (calcEnergy) {
      cudaMemcpy(energy,energy_d,eeend*sizeof(real),cudaMemcpyDeviceToHost);
    }
  }
  MPI_Gather(forceBuffer,N,MPI_CREAL,forceBuffer,N,MPI_CREAL,0,MPI_COMM_WORLD); // NOMPI
  if (calcEnergy) {
    MPI_Gather(energy,eeend,MPI_CREAL,energy,eeend,MPI_CREAL,0,MPI_COMM_WORLD); // NOMPI
  }
  if (system->id==0) {
    cudaMemcpy(forceBuffer_d+N,forceBuffer+N,(system->idCount-1)*N*sizeof(real),cudaMemcpyHostToDevice);
    if (calcEnergy) {
      cudaMemcpy(energy_d+eeend,energy+eeend,(system->idCount-1)*eeend*sizeof(real),cudaMemcpyHostToDevice);
    }
  }*/ // NOMPI
// #define ORIGINAL_GATHER
#ifdef ORIGINAL_GATHER
#pragma omp barrier // OMP
  if (system->id!=0) { // OMP
    // cudaMemcpyPeer(forceBuffer_omp,0,forceBuffer_d,system->id,N*sizeof(real)); // OMP
    cudaMemcpy(forceBuffer_omp,forceBuffer_d,N*sizeof(real_f),cudaMemcpyDefault); // OMP
    if (calcEnergy) { // OMP
      cudaMemcpy(energy_omp,energy_d,eeend*sizeof(real_e),cudaMemcpyDefault); // OMP
    } // OMP
  } // OMP
#pragma omp barrier // OMP
  // nvtxRangePop();
#else
  if (system->id!=0) {
    // cudaMemcpyAsync(forceBuffer_omp,forceBuffer_d,N*sizeof(real_f),cudaMemcpyDefault,system->run->updateStream);
    cudaMemcpyPeerAsync(forceBuffer_omp,0,forceBuffer_d,system->id,N*sizeof(real_f),system->run->updateStream);
    if (calcEnergy) {
      // cudaMemcpyAsync(energy_omp,energy_d,eeend*sizeof(real_e),cudaMemcpyDefault,system->run->updateStream);
      cudaMemcpyPeerAsync(energy_omp,0,energy_d,system->id,eeend*sizeof(real_e),system->run->updateStream);
    }
  }
  /* // Binary wait tree
  for (int i=0; (1<<i)<system->idCount; i++) {
    int eligible=(((system->id)&((1<<i)-1))==0);
    int send=((system->id)&(1<<i)); // senders equal 1<<i, receivers equal 0
    int partner=((system->id)^(1<<i));
    if (eligible && send) {
      // fprintf(stdout,"Step %d id %d sending to partner %d\n",system->run->step,system->id,partner);
      cudaEventRecord(system->run->communicate,system->run->updateStream);
    }
#pragma omp barrier
    if (eligible && (!send) && partner<system->idCount) {
      // fprintf(stdout,"Step %d id %d receiving from partner %d\n",system->run->step,system->id,partner);
      cudaStreamWaitEvent(system->run->updateStream,system->run->communicate_omp[partner],0);
    }
  } */
  // All to one naive wait - if there are enough nodes for the tree to matter, the copy, not the synchronization will be rate limiting. The tree structure may take longer anyways because more events have to be serially recorded.
  if (system->id!=0) {
    cudaEventRecord(system->run->communicate,system->run->updateStream);
  }
#pragma omp barrier
  if (system->id==0) {
    for (int i=1; i<system->idCount; i++) {
      cudaStreamWaitEvent(system->run->updateStream,system->run->communicate_omp[i],0);
    }
  }
#endif
}

void State::prettify_position(System *system)
{
  int i,j,k;
  real3_x pos, ref;
  if (system->run->prettyXTC) {
    for (i=0; i<atomCount; i++) {
      j=system->potential->prettifyPlan[i][0];
      k=system->potential->prettifyPlan[i][1];
      if (k<0) { // Put it in the box
        ref.x=0;
        ref.y=0;
        ref.z=0;
      } else { // Put it next to particle k
        ref=((real3_x*)position)[k];
      }
      pos=((real3_x*)position)[j];
      if (typeBox) {
        real_x shift;
        shift=floor((ref.z-pos.z)/tricBox.c.z+((real_x)0.5));
        pos.x+=tricBox.c.x*shift;
        pos.y+=tricBox.c.y*shift;
        pos.z+=tricBox.c.z*shift;
        shift=floor((ref.y-pos.y)/tricBox.b.y+((real_x)0.5));
        pos.x+=tricBox.b.x*shift;
        pos.y+=tricBox.b.y*shift;
        shift=floor((ref.x-pos.x)/tricBox.a.x+((real_x)0.5));
        pos.x+=tricBox.a.x*shift;
      } else {
        pos.x+=orthBox.x*floor((ref.x-pos.x)/orthBox.x+((real_x)0.5));
        pos.y+=orthBox.y*floor((ref.y-pos.y)/orthBox.y+((real_x)0.5));
        pos.z+=orthBox.z*floor((ref.z-pos.z)/orthBox.z+((real_x)0.5));
      }
      ((real3_x*)position)[j]=pos;
    }
  }
}

void State::check_box(System *system)
{
  // Check angles
  if (nameBox==ebcubi || nameBox==ebtetr || nameBox==eborth) {
    if (box.b.x!=90 || box.b.y!=90 || box.b.z!=90) {
      fprintf(stdout,"Warning: cubic, tetragonal, or orthorhombic box does not have all 90 degree angles\n");
      fprintf(stdout,"Previous: alpha %24.16f beta %24.16f gamma %24.16f\n",box.b.x,box.b.y,box.b.z);
      box.b.x=90;
      box.b.y=90;
      box.b.z=90;
      fprintf(stdout,"Rectified: alpha %24.16f beta %24.16f gamma %24.16f\n",box.b.x,box.b.y,box.b.z);
    }
  } else if (nameBox==ebmono) {
    if (box.b.x!=90 || box.b.z!=90) {
      fprintf(stdout,"Warning: monoclinic box must have alpha and gamma equal to 90 degrees\n");
      fprintf(stdout,"Previous: alpha %24.16f beta %24.16f gamma %24.16f\n",box.b.x,box.b.y,box.b.z);
      box.b.x=90;
      box.b.z=90;
      fprintf(stdout,"Rectified: alpha %24.16f beta %24.16f gamma %24.16f\n",box.b.x,box.b.y,box.b.z);
    }
  } else if (nameBox==ebhexa) {
    if (box.b.x!=90 || box.b.y!=90 || box.b.z!=120) {
      fprintf(stdout,"Warning: hexagonal box must have 90, 90, 120 degree angles in that order\n");
      fprintf(stdout,"Previous: alpha %24.16f beta %24.16f gamma %24.16f\n",box.b.x,box.b.y,box.b.z);
      box.b.x=90;
      box.b.y=90;
      box.b.z=120;
      fprintf(stdout,"Rectified: alpha %24.16f beta %24.16f gamma %24.16f\n",box.b.x,box.b.y,box.b.z);
    }
  } else if (nameBox==ebocta) {
    // DEGREES macro is only floating precision, not double as needed here
    real_x alpha=acos(-1./3.)/0.017453292519943295769;
    if (box.b.x!=alpha || box.b.y!=alpha || box.b.z!=alpha) {
      fprintf(stdout,"Warning: octahedral box must have 109.471220634 degree angles\n");
      fprintf(stdout,"Previous: alpha %24.16f beta %24.16f gamma %24.16f\n",box.b.x,box.b.y,box.b.z);
      box.b.x=alpha;
      box.b.y=alpha;
      box.b.z=alpha;
      fprintf(stdout,"Rectified: alpha %24.16f beta %24.16f gamma %24.16f\n",box.b.x,box.b.y,box.b.z);
    }
  } else if (nameBox==ebrhdo) {
    if (box.b.x!=60 || box.b.y!=90 || box.b.z!=60) {
      fprintf(stdout,"Warning: rhombic dodecahedron box must have 60, 90, 60 degree angles in that order\n");
      fprintf(stdout,"Previous: alpha %24.16f beta %24.16f gamma %24.16f\n",box.b.x,box.b.y,box.b.z);
      box.b.x=60;
      box.b.y=90;
      box.b.z=60;
      fprintf(stdout,"Rectified: alpha %24.16f beta %24.16f gamma %24.16f\n",box.b.x,box.b.y,box.b.z);
    }
  }
  // Check lengths
  if (nameBox==ebcubi || nameBox==ebrhom || nameBox==ebocta || nameBox==ebrhdo) {
    if (box.a.x!=box.a.y || box.a.x!=box.a.z) {
      fprintf(stdout,"Warning: all three box vectors must have the same length for cubic, rhombohedral, trucated octahedron, and rhombic dodecahedron boxes\n");
      fprintf(stdout,"Previous: a %24.16f b %24.16f c %24.16f\n",box.a.x,box.a.y,box.a.z);
      box.a.y=box.a.x;
      box.a.z=box.a.x;
      fprintf(stdout,"Rectified: a %24.16f b %24.16f c %24.16f\n",box.a.x,box.a.y,box.a.z);
    }
  } else if (nameBox==ebtetr || nameBox==ebhexa) {
    if (box.a.x!=box.a.y) {
      fprintf(stdout,"Warning: first two box vectors must have the same length for tetragonal and hexagonal boxes\n");
      fprintf(stdout,"Previous: a %24.16f b %24.16f c %24.16f\n",box.a.x,box.a.y,box.a.z);
      box.a.y=box.a.x;
      fprintf(stdout,"Rectified: a %24.16f b %24.16f c %24.16f\n",box.a.x,box.a.y,box.a.z);
    }
  }

  broadcast_box(system);

  if (nameBox!=ebcubi && nameBox!=ebtetr && nameBox!=eborth) {
    if (abs(tricBox.b.x)>tricBox.a.x*0.5000005 || abs(tricBox.c.x)>tricBox.a.x*0.5000005 || abs(tricBox.c.y)>tricBox.b.y*0.5000005) {
      fatal(__FILE__,__LINE__,"Fatal, non-orthogonal box must be in a minimal lattice near orthogonal for the PBC assumptions underlying BLaDE to hold.\n");
    }
  }
}
