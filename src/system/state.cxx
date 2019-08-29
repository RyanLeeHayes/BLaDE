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
  int i,j;

  atomCount=system->coordinates->atomCount;
  lambdaCount=system->msld->blockCount;

  int n=atomCount;
  int nL=lambdaCount;
  int rootFactor=(system->id==0?system->idCount:1);

  // Buffers for transfer and reduction

  // Lambda-Spatial-Theta buffers
  positionBuffer=(real*)calloc((2*nL+3*n),sizeof(real));
  cudaMalloc(&(positionBuffer_d),(2*nL+3*n)*sizeof(real));
  cudaMalloc(&(positionBackup_d),(2*nL+3*n)*sizeof(real));
  forceBuffer=(real*)calloc(rootFactor*(2*nL+3*n),sizeof(real));
  cudaMalloc(&(forceBuffer_d),rootFactor*(2*nL+3*n)*sizeof(real));
  cudaMalloc(&(forceBackup_d),(2*nL+3*n)*sizeof(real));

  if (system->idCount>0) { // OMP
#pragma omp barrier // OMP
    if (system->id==0) { // OMP
      system->message[0]=(void*)positionBuffer_d; // OMP
      for (i=1; i<system->idCount; i++) { // OMP
        system->message[i]=(void*)&forceBuffer_d[i*(2*nL+3*n)]; // OMP
      } // OMP
#pragma omp barrier // master barrier // OMP
    } else { // OMP
#pragma omp barrier // everyone else's barrier // OMP
      positionBuffer_omp=(real*)(system->message[0]); // OMP
      forceBuffer_omp=(real*)(system->message[system->id]); // OMP
    } // OMP
#pragma omp barrier // OMP
  } // OMP

  // Other buffers
  energy=(real*)calloc(rootFactor*eeend,sizeof(real));
  cudaMalloc(&(energy_d),rootFactor*eeend*sizeof(real));
  cudaMalloc(&(energyBackup_d),eeend*sizeof(real));

  if (system->idCount>0) { // OMP
#pragma omp barrier // OMP
    if (system->id==0) { // OMP
      system->message[0]=(void*)&orthBox; // OMP
      for (i=1; i<system->idCount; i++) { // OMP
        system->message[i]=(void*)&energy_d[i*eeend]; // OMP
      } // OMP
#pragma omp barrier // master barrier // OMP
    } else { // OMP
#pragma omp barrier // everyone else's barrier // OMP
      orthBox_omp=(real3*)(system->message[0]); // OMP
      energy_omp=(real*)(system->message[system->id]); // OMP
    } // OMP
#pragma omp barrier // OMP
  } // OMP

  // Spatial-Theta buffers
  velocityBuffer=(real*)calloc((nL+3*n),sizeof(real));
  cudaMalloc(&(velocityBuffer_d),(nL+3*n)*sizeof(real));
  invsqrtMassBuffer=(real*)calloc((nL+3*n),sizeof(real));
  cudaMalloc(&(invsqrtMassBuffer_d),(nL+3*n)*sizeof(real));

  // Constraint stuff
  positionCons_d=NULL;
  if (system->structure->shakeHbond) {
    cudaMalloc(&(positionCons_d),(nL+3*n)*sizeof(real));
  }

  // The box
  orthBox=system->coordinates->particleOrthBox;

  // Buffer for floating point output
  positionXTC=(float(*)[3])calloc(n,sizeof(float[3])); // intentional float

  // Labels (do not free)
  lambda=positionBuffer;
  lambda_d=positionBuffer_d;
  position=(real(*)[3])(positionBuffer+nL);
  position_d=(real(*)[3])(positionBuffer_d+nL);
  theta=positionBuffer+nL+3*n;
  theta_d=positionBuffer_d+nL+3*n;
  velocity=(real(*)[3])velocityBuffer;
  velocity_d=(real(*)[3])velocityBuffer_d;
  thetaVelocity=velocityBuffer+3*n;
  thetaVelocity_d=velocityBuffer_d+3*n;
  lambdaForce=forceBuffer;
  lambdaForce_d=forceBuffer_d;
  force=(real(*)[3])(forceBuffer+nL);
  force_d=(real(*)[3])(forceBuffer_d+nL);
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
  leapState=alloc_leapstate(3*atomCount,lambdaCount,(real*)position_d,velocityBuffer_d,(real*)force_d,invsqrtMassBuffer_d);
}

State::~State() {
  // Lambda-Spatial-Theta buffers
  if (positionBuffer) free(positionBuffer);
  if (positionBuffer_d) cudaFree(positionBuffer_d);
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

  cudaMemcpy(positionBuffer_d,positionBuffer,(2*nL+3*n)*sizeof(real),cudaMemcpyHostToDevice);
  cudaMemcpy(velocityBuffer_d,velocityBuffer,(nL+3*n)*sizeof(real),cudaMemcpyHostToDevice);
  cudaMemset(forceBuffer_d,0,(nL+3*n)*sizeof(real));
  cudaMemcpy(invsqrtMassBuffer_d,invsqrtMassBuffer,(nL+3*n)*sizeof(real),cudaMemcpyHostToDevice);
  system->msld->calc_lambda_from_theta(0,system);
}



struct LeapParms1* State::alloc_leapparms1(real dt,real gamma,real T)
{
  struct LeapParms1 *lp;
  real kT=kB*T;

  lp=(struct LeapParms1*) malloc(sizeof(struct LeapParms1));

  // Integrator from https://pubs.acs.org/doi/10.1021/jp411770f
  lp->dt=dt;
  lp->gamma=gamma;
  lp->kT=kT;

  return lp;
}

struct LeapParms2* State::alloc_leapparms2(real dt,real gamma,real T)
{
  struct LeapParms2 *lp;
  real kT=kB*T;
  real a=exp(-gamma*dt);
  real b=sqrt(tanh(0.5*gamma*dt)/(0.5*gamma*dt));

  lp=(struct LeapParms2*) malloc(sizeof(struct LeapParms2));

  // Integrator from https://pubs.acs.org/doi/10.1021/jp411770f
  lp->sqrta=sqrt(a);
  lp->noise=sqrt((1-a)*kT);
  lp->fscale=0.5*b*dt;

  return lp;
}

struct LeapState* State::alloc_leapstate(int N1,int N2,real *x,real *v,real *f,real *ism)
{
  struct LeapState *ls;

  ls=(struct LeapState*) malloc(sizeof(struct LeapState));
  real *random;
  cudaMalloc(&random,2*(N1+N2)*sizeof(real));

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
  cudaMemcpy(theta,theta_d,lambdaCount*sizeof(real),cudaMemcpyDeviceToHost);
  cudaMemcpy(position,position_d,3*atomCount*sizeof(real),cudaMemcpyDeviceToHost);
  cudaMemcpy(thetaVelocity,thetaVelocity_d,lambdaCount*sizeof(real),cudaMemcpyDeviceToHost);
  cudaMemcpy(velocity,velocity_d,3*atomCount*sizeof(real),cudaMemcpyDeviceToHost);
}

void State::send_state()
{
  cudaMemcpy(theta_d,theta,lambdaCount*sizeof(real),cudaMemcpyHostToDevice);
  cudaMemcpy(position_d,position,3*atomCount*sizeof(real),cudaMemcpyHostToDevice);
  cudaMemcpy(thetaVelocity_d,thetaVelocity,lambdaCount*sizeof(real),cudaMemcpyHostToDevice);
  cudaMemcpy(velocity_d,velocity,3*atomCount*sizeof(real),cudaMemcpyHostToDevice);
}

void State::recv_position()
{
  cudaMemcpy(position,position_d,3*atomCount*sizeof(real),cudaMemcpyDeviceToHost);
}

void State::recv_lambda()
{
  cudaMemcpy(lambda,lambda_d,lambdaCount*sizeof(real),cudaMemcpyDeviceToHost);
}

void State::recv_energy()
{
  int i;

  cudaMemcpy(energy,energy_d,eeend*sizeof(real),cudaMemcpyDeviceToHost);

  for (i=0; i<eepotential; i++) {
    energy[eepotential]+=energy[i];
  }
  energy[eetotal]=energy[eepotential]+energy[eekinetic];
}


void State::backup_position()
{
  cudaMemcpy(positionBackup_d,positionBuffer_d,(2*lambdaCount+3*atomCount)*sizeof(real),cudaMemcpyDeviceToDevice);
  cudaMemcpy(forceBackup_d,forceBuffer_d,(2*lambdaCount+3*atomCount)*sizeof(real),cudaMemcpyDeviceToDevice);
  cudaMemcpy(energyBackup_d,energy_d,eeend*sizeof(real),cudaMemcpyDeviceToDevice);
  orthBoxBackup=orthBox;
}

void State::restore_position()
{
  cudaMemcpy(positionBuffer_d,positionBackup_d,(2*lambdaCount+3*atomCount)*sizeof(real),cudaMemcpyDeviceToDevice);
  cudaMemcpy(forceBuffer_d,forceBackup_d,(2*lambdaCount+3*atomCount)*sizeof(real),cudaMemcpyDeviceToDevice);
  cudaMemcpy(energy_d,energyBackup_d,eeend*sizeof(real),cudaMemcpyDeviceToDevice);
  orthBox=orthBoxBackup;
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
    cudaEventRecord(system->run->communicate,system->run->updateStream);
#pragma omp barrier
  } else { // OMP
#pragma omp barrier
    cudaEventSynchronize(system->run->communicate_omp[0]);
    // cudaMemcpyPeer(positionBuffer_d,system->id,positionBuffer_omp,0,N*sizeof(real)); // OMP
    cudaMemcpyAsync(positionBuffer_d,positionBuffer_omp,N*sizeof(real),cudaMemcpyDefault,system->run->updateStream); // OMP
  } // OMP
  // nvtxRangePop();
}

void State::broadcast_box(System *system)
{
  // int N=3; // NOMPI
  // MPI_Bcast(&orthBox,N,MPI_CREAL,0,MPI_COMM_WORLD); // NOMPI
  // nvtxRangePushA("broadcast_box");
#pragma omp barrier // OMP
  if (system->id!=0) { // OMP
    orthBox=orthBox_omp[0]; // OMP
  } // OMP
#pragma omp barrier // OMP
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
#pragma omp barrier // OMP
  if (system->id!=0) { // OMP
    // cudaMemcpyPeer(forceBuffer_omp,0,forceBuffer_d,system->id,N*sizeof(real)); // OMP
    cudaMemcpy(forceBuffer_omp,forceBuffer_d,N*sizeof(real),cudaMemcpyDefault); // OMP
    if (calcEnergy) { // OMP
      cudaMemcpy(energy_omp,energy_d,eeend*sizeof(real),cudaMemcpyDefault); // OMP
    } // OMP
  } // OMP
#pragma omp barrier // OMP
  // nvtxRangePop();
}
