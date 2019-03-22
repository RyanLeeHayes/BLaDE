#include <cuda_runtime.h>
#include <mpi.h>

#include "domdec/domdec.h"
#include "system/system.h"
#include "system/state.h"
#include "run/run.h"
#include "main/defines.h"
#include "main/real3.h"



__global__ void assign_domain_kernel(int atomCount,real3 *position,real3 box,int3 gridDomdec,int *domain)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real3 xi;
  int3 idDomdec;

  if (i<atomCount) {
    xi=position[i];
    xi=real3_modulus(xi,box);
    position[i]=xi;

    idDomdec.x=(int) floor(xi.x*gridDomdec.x/box.x);
    idDomdec.x-=(idDomdec.x>=gridDomdec.x?gridDomdec.x:0);
    idDomdec.y=(int) floor(xi.y*gridDomdec.y/box.y);
    idDomdec.y-=(idDomdec.y>=gridDomdec.y?gridDomdec.y:0);
    idDomdec.z=(int) floor(xi.z*gridDomdec.z/box.z);
    idDomdec.z-=(idDomdec.z>=gridDomdec.z?gridDomdec.z:0);

    domain[i]=(idDomdec.x*gridDomdec.y+idDomdec.y)*gridDomdec.z+idDomdec.z;
  }
}

void Domdec::broadcast_domain(System *system)
{
  int N=globalCount;
  if (system->id==0) {
    cudaMemcpy(domain,domain_d,N*sizeof(int),cudaMemcpyDeviceToHost);
  }
  MPI_Bcast(domain,N,MPI_INT,0,MPI_COMM_WORLD);
  if (system->id!=0) {
    cudaMemcpy(domain_d,domain,N*sizeof(int),cudaMemcpyHostToDevice);
  }
}

void Domdec::assign_domain(System *system)
{
  Run *r=system->run;
  if (system->id==0) {
    assign_domain_kernel<<<(system->state->atomCount+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(system->state->atomCount,(real3*)system->state->position_d,system->state->orthBox,gridDomdec,domain_d);
  }

  if (system->idCount!=1) {
    broadcast_domain(system);
    system->state->broadcast_position(system);
  }
}
