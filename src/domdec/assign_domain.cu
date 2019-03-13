#include <cuda_runtime.h>
#include <mpi.h>

#include "domdec/domdec.h"
#include "system/system.h"
#include "system/state.h"
#include "update/update.h"
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

void Domdec::assign_domain(System *system)
{
  if (system->id==0) {
    assign_domain_kernel<<<(system->state->atomCount+BLUP-1)/BLUP,BLUP,0,system->update->updateStream>>>(system->state->atomCount,(real3*)system->state->position_d,system->state->orthBox,gridDomdec,domain_d);
  }
return;
  if (system->idCount!=1) {
    MPI_Bcast(domain_d,system->state->atomCount,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(system->state->position_d,3*system->state->atomCount,MPI_FLOAT,0,MPI_COMM_WORLD);
  }
}
