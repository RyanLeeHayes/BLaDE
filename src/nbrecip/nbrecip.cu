#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>

#include "nbrecip/nbrecip.h"
#include "main/defines.h"
#include "system/system.h"
#include "msld/msld.h"
#include "system/state.h"
#include "run/run.h"
#include "system/potential.h"

#include "main/real3.h"



// getforce_ewaldself_kernel<<<(N+BLNB-1)/BLNB,BLNB,shMem,p->bondedStream>>>(N,p->charge_d,prefactor,m->atomBlock_d,m->lambda_d,m->lambdaForce_d,pEnergy);
__global__ void getforce_ewaldself_kernel(int atomCount,real *charge,real prefactor,int *atomBlock,real *lambda,real *lambdaForce,real *energy)
{
// NYI - maybe energy should be a double
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real q;
  real lEnergy=0;
  extern __shared__ real sEnergy[];
  int b;
  real l=1;
  
  if (i<atomCount) {
    q=charge[i];
    
    // Scaling
    b=atomBlock[i];
    if (b) {
      l=lambda[b];
    }

    // interaction
    if (b || energy) {
      lEnergy=prefactor*q*q;
    }

    // Lambda force
    if (b) {
      realAtomicAdd(&lambdaForce[b],2*l*lEnergy);
    }
  }

  // Energy, if requested
  if (energy) {
    lEnergy*=l*l;
    real_sum_reduce(lEnergy,sEnergy,energy);
  }
}

void getforce_ewaldself(System *system,bool calcEnergy)
{
  Potential *p=system->potential;
  State *s=system->state;
  Msld *m=system->msld;
  Run *r=system->run;
  int N=p->atomCount;
  int shMem=0;
  real *pEnergy=NULL;
  real prefactor=-system->run->betaEwald*(kELECTRIC/sqrt(M_PI));

  if (calcEnergy) {
    shMem=BLNB*sizeof(real)/32;
    pEnergy=s->energy_d+eenbrecipself;
  }

  getforce_ewaldself_kernel<<<(N+BLNB-1)/BLNB,BLNB,shMem,r->nbrecipStream>>>(N,p->charge_d,prefactor,m->atomBlock_d,s->lambda_d,s->lambdaForce_d,pEnergy);
}



// getforce_ewald_spread_kernel<<<>>>(N,p->charge_d,m->atomBlock_d,(real3*)s->position_d,s->orthBox,m->lambda_d,((int3*)p->gridDimPME)[0],p->chargeGridPME_d);
__global__ void getforce_ewald_spread_kernel(int atomCount,real *charge,int *atomBlock,real3* position,real3 box,real *lambda,int3 gridDimPME,real *chargeGridPME)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real q;
  int b;
  real l=1;
  int order=4;
  real3 xi; // position
  real u; // fractional coordinate remainder
  int3 u0; // index of grid point
  real M2,M3,M4; // 2nd and 4th order B splines
  real3 density;
  // real3 dDensity;
  real2 dIndex;
  int j,k;
  int index;
  int masterThread=threadIdx.x&(-4u);

  if (i/order<atomCount) {
    q=charge[i/order];

    // Scaling
    b=atomBlock[i/order];
    if (b) {
      l=lambda[b];
    }
    q*=l;

    xi=position[i/order];

    // Get grid position
    u=xi.x*gridDimPME.x/box.x;
    u0.x=(int)floor(u);
    u-=u0.x;
    // u0.x=(u0.x-3)%gridDimPME.x;
    u0.x=u0.x%gridDimPME.x;
    u0.x+=(u0.x<0?gridDimPME.x:0);
// Required due to modulus error
    u0.x-=(u0.x>=gridDimPME.x?gridDimPME.x:0);
    u+=(masterThread+3-threadIdx.x); // very important 

    M2=((i&3)==3)?u:0;
    M2=((i&3)==2)?2-u:M2; // Second order B spline
    M3=u*M2+(3-u)*__shfl_sync(0xFFFFFFFF,M2,i+1,order);
    M3*=0.5f; // 1/(n-1)
    M4=u*M3+(4-u)*__shfl_sync(0xFFFFFFFF,M3,i+1,order);
    M4*=0.333333333333333333f; // 1/(n-1)
    density.x=M4;
    // dDensity.x=M3-__shfl_sync(0xFFFFFFFF,M3,i+1,order);

    u=xi.y*gridDimPME.y/box.y;
    u0.y=(int)floor(u);
    u-=u0.y;
    // u0.y=(u0.y-3)%gridDimPME.y;
    u0.y=u0.y%gridDimPME.y;
    u0.y+=(u0.y<0?gridDimPME.y:0);
    u0.y-=(u0.y>=gridDimPME.y?gridDimPME.y:0);
    u+=(masterThread+3-threadIdx.x); // very important 

    M2=((i&3)==3)?u:0;
    M2=((i&3)==2)?2-u:M2; // Second order B spline
    M3=u*M2+(3-u)*__shfl_sync(0xFFFFFFFF,M2,i+1,order);
    M3*=0.5f; // 1/(n-1)
    M4=u*M3+(4-u)*__shfl_sync(0xFFFFFFFF,M3,i+1,order);
    M4*=0.333333333333333333f; // 1/(n-1)
    density.y=M4;
    // dDensity.y=M3-__shfl_sync(0xFFFFFFFF,M3,i+1,order);

    u=xi.z*gridDimPME.z/box.z;
    u0.z=(int)floor(u);
    u-=u0.z;
    // u0.z=(u0.z-3)%gridDimPME.z;
    u0.z=u0.z%gridDimPME.z;
    u0.z+=(u0.z<0?gridDimPME.z:0);
    u0.z-=(u0.z>=gridDimPME.z?gridDimPME.z:0);
    u+=(masterThread+3-threadIdx.x); // very important 

    M2=((i&3)==3)?u:0;
    M2=((i&3)==2)?2-u:M2; // Second order B spline
    M3=u*M2+(3-u)*__shfl_sync(0xFFFFFFFF,M2,i+1,order);
    M3*=0.5f; // 1/(n-1)
    M4=u*M3+(4-u)*__shfl_sync(0xFFFFFFFF,M3,i+1,order);
    M4*=0.333333333333333333f; // 1/(n-1)
    density.z=M4;
    // dDensity.z=M3-__shfl_sync(0xFFFFFFFF,M3,i+1,order);

    for (j=0; j<order; j++) {
      dIndex.x=__shfl_sync(0xFFFFFFFF,density.x,masterThread+j);
      for (k=0; k<order; k++) {
        dIndex.y=__shfl_sync(0xFFFFFFFF,density.y,masterThread+k);
        index=((u0.x+j)%gridDimPME.x);
        index*=gridDimPME.y;
        index+=((u0.y+k)%gridDimPME.y);
        index*=gridDimPME.z;
        index+=((u0.z+threadIdx.x-masterThread)%gridDimPME.z);
        realAtomicAdd(&chargeGridPME[index],q*dIndex.x*dIndex.y*density.z);
      }
    }
  }
}

// getforce_ewald_convolution_kernel<<<blockCount,blockSize,0,p->nbrecipStream>>>(((int3*)gridDimPME)[0],p->fourierGridPME_d,p->bGridPME_d,system->run->betaEwald,s->orthoBox)
__global__ void getforce_ewald_convolution_kernel(int3 gridDimPME,cufftComplex *fourierGridPME,real *bGridPME,real betaEwald,real3 box)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int j=blockIdx.y*blockDim.y+threadIdx.y;
  int k=blockIdx.z*blockDim.z+threadIdx.z;
  int ijk=((i*gridDimPME.y)+j)*(gridDimPME.z/2+1)+k;
  real V=box.x*box.y*box.z;
  real kcomp;
  real k2; // squared reciprocal space vector
  real factor;

  if (i<gridDimPME.x && j<gridDimPME.y && k<(gridDimPME.z/2+1)) {
    kcomp=(2*i<=gridDimPME.x?i:i-gridDimPME.x)/box.x;
    k2=kcomp*kcomp;
    kcomp=(2*j<=gridDimPME.y?j:j-gridDimPME.y)/box.y;
    k2+=kcomp*kcomp;
    kcomp=k/box.z; // always true // (2*k<=gridDimPME.z?k:k-gridDimPME.z)/box.z;
    k2+=kcomp*kcomp;
    factor=bGridPME[ijk];
    factor*=(0.5*kELECTRIC/M_PI)/V;
    // factor/=(gridDimPME.x*gridDimPME.y*gridDimPME.z); // cufft Normalization
    factor*=exp(-M_PI*M_PI*k2/(betaEwald*betaEwald));
    factor/=k2;
    // factor*=(2*i==gridDimPME.x?0:1); // Oops, looks like edges are counted once, not twice in paper
    // factor*=(2*j==gridDimPME.y?0:1);
    // factor*=(2*k==gridDimPME.z?0:1);
    factor=(ijk==0?0:factor);
    fourierGridPME[ijk].x*=factor;
    fourierGridPME[ijk].y*=factor;
  }
}

/*
__global__ void getforce_ewald_regularize_kernel(int3 gridDimPME,cufftComplex *fourierGridPME)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int j=blockIdx.y*blockDim.y+threadIdx.y;
  int k=(gridDimPME.z/2)*threadIdx.z;
  int nmi,nmj;
  cufftComplex mean;

  nmi=gridDimPME.x-i;
  nmi-=(nmi>=gridDimPME.x?gridDimPME.x:0); // modulus
  nmj=gridDimPME.y-j;
  nmj-=(nmj>=gridDimPME.y?gridDimPME.y:0); // modulus
  if (i<gridDimPME.x && j<(gridDimPME.x/2+1)) {
    mean.x=fourierGridPME[(i*gridDimPME.y+j)*(gridDimPME.z/2+1)+k].x;
    mean.y=fourierGridPME[(i*gridDimPME.y+j)*(gridDimPME.z/2+1)+k].y;
    mean.x+=fourierGridPME[(nmi*gridDimPME.y+nmj)*(gridDimPME.z/2+1)+k].x;
    mean.y-=fourierGridPME[(nmi*gridDimPME.y+nmj)*(gridDimPME.z/2+1)+k].y;
    mean.x*=0.5f;
    mean.y*=0.5f;
    fourierGridPME[(i*gridDimPME.y+j)*(gridDimPME.z/2+1)+k].x=mean.x;
    fourierGridPME[(i*gridDimPME.y+j)*(gridDimPME.z/2+1)+k].y=mean.y;
    fourierGridPME[(nmi*gridDimPME.y+nmj)*(gridDimPME.z/2+1)+k].x=mean.x;
    fourierGridPME[(nmi*gridDimPME.y+nmj)*(gridDimPME.z/2+1)+k].y=-mean.y;
  }
}
*/

// getforce_ewald_gather_kernel<<<>>>(N,p->charge_d,prefactor,m->atomBlock_d,((int3*)p->gridDimPME)[0],p->potentialGridPME_d,(real3*)s->position_d,(real3*)s->force_d,s->orthBox,m->lambda_d,m->lambdaForce_d,pEnergy);
__global__ void getforce_ewald_gather_kernel(int atomCount,real *charge,int *atomBlock,int3 gridDimPME,real *potentialGridPME,real3 *position,real3 *force,real3 box,real *lambda,real *lambdaForce,real *energy)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real q;
  int b;
  real l=1;
  int order=4;
  real3 xi; // position
  real3 fi;
  real u; // fractional coordinate remainder
  int3 u0; // index of grid point
  real M2,M3,M4; // 2nd and 4th order B splines
  real3 density;
  real3 dDensity;
  real2 dIndex, dDIndex;
  int j,k;
  int index;
  int masterThread=threadIdx.x&(-4u);
  real lEnergy=0;
  extern __shared__ real sEnergy[];

  if (i/order<atomCount) {
    q=charge[i/order];

    // Scaling
    b=atomBlock[i/order];
    if (b) {
      l=lambda[b];
    }
    // q*=l; // do this scaling later in the kernel

    xi=position[i/order];

    // Get grid position
    u=xi.x*gridDimPME.x/box.x;
    u0.x=(int)floor(u);
    u-=u0.x;
    // u0.x=(u0.x-3)%gridDimPME.x;
    u0.x=u0.x%gridDimPME.x;
    u0.x+=(u0.x<0?gridDimPME.x:0);
    u0.x-=(u0.x>=gridDimPME.x?gridDimPME.x:0);
    u+=(masterThread+3-threadIdx.x); // very important 

    M2=((i&3)==3)?u:0;
    M2=((i&3)==2)?2-u:M2; // Second order B spline
    M3=u*M2+(3-u)*__shfl_sync(0xFFFFFFFF,M2,i+1,order);
    M3*=0.5f; // 1/(n-1)
    M4=u*M3+(4-u)*__shfl_sync(0xFFFFFFFF,M3,i+1,order);
    M4*=0.333333333333333333f; // 1/(n-1)
    density.x=M4;
    dDensity.x=M3-__shfl_sync(0xFFFFFFFF,M3,i+1,order);

    u=xi.y*gridDimPME.y/box.y;
    u0.y=(int)floor(u);
    u-=u0.y;
    // u0.y=(u0.y-3)%gridDimPME.y;
    u0.y=u0.y%gridDimPME.y;
    u0.y+=(u0.y<0?gridDimPME.y:0);
    u0.y-=(u0.y>=gridDimPME.y?gridDimPME.y:0);
    u+=(masterThread+3-threadIdx.x); // very important 

    M2=((i&3)==3)?u:0;
    M2=((i&3)==2)?2-u:M2; // Second order B spline
    M3=u*M2+(3-u)*__shfl_sync(0xFFFFFFFF,M2,i+1,order);
    M3*=0.5f; // 1/(n-1)
    M4=u*M3+(4-u)*__shfl_sync(0xFFFFFFFF,M3,i+1,order);
    M4*=0.333333333333333333f; // 1/(n-1)
    density.y=M4;
    dDensity.y=M3-__shfl_sync(0xFFFFFFFF,M3,i+1,order);

    u=xi.z*gridDimPME.z/box.z;
    u0.z=(int)floor(u);
    u-=u0.z;
    // u0.z=(u0.z-3)%gridDimPME.z;
    u0.z=u0.z%gridDimPME.z;
    u0.z+=(u0.z<0?gridDimPME.z:0);
    u0.z-=(u0.z>=gridDimPME.z?gridDimPME.z:0);
    u+=(masterThread+3-threadIdx.x); // very important 

    M2=((i&3)==3)?u:0;
    M2=((i&3)==2)?2-u:M2; // Second order B spline
    M3=u*M2+(3-u)*__shfl_sync(0xFFFFFFFF,M2,i+1,order);
    M3*=0.5f; // 1/(n-1)
    M4=u*M3+(4-u)*__shfl_sync(0xFFFFFFFF,M3,i+1,order);
    M4*=0.333333333333333333f; // 1/(n-1)
    density.z=M4;
    dDensity.z=M3-__shfl_sync(0xFFFFFFFF,M3,i+1,order);

    // Everything is the same as the spread kernel up to this point
    fi.x=0;
    fi.y=0;
    fi.z=0;
    for (j=0; j<order; j++) {
      dIndex.x=__shfl_sync(0xFFFFFFFF,density.x,masterThread+j);
      dDIndex.x=__shfl_sync(0xFFFFFFFF,dDensity.x,masterThread+j);
      for (k=0; k<order; k++) {
        dIndex.y=__shfl_sync(0xFFFFFFFF,density.y,masterThread+k);
        dDIndex.y=__shfl_sync(0xFFFFFFFF,dDensity.y,masterThread+k);
        index=((u0.x+j)%gridDimPME.x);
        index*=gridDimPME.y;
        index+=((u0.y+k)%gridDimPME.y);
        index*=gridDimPME.z;
        index+=((u0.z+threadIdx.x-masterThread)%gridDimPME.z);
        fi.x+=potentialGridPME[index]*dDIndex.x*dIndex.y*density.z;
        fi.y+=potentialGridPME[index]*dIndex.x*dDIndex.y*density.z;
        fi.z+=potentialGridPME[index]*dIndex.x*dIndex.y*dDensity.z;
        if (b || energy) {
          lEnergy+=potentialGridPME[index]*dIndex.x*dIndex.y*density.z;
        }
      }
    }

    // Lambda force
    if (b) {
#warning "4 threads trying together"
      realAtomicAdd(&lambdaForce[b],q*lEnergy);
    }

    // Spatial force
    fi.x*=l*q*gridDimPME.x/box.x;
    fi.y*=l*q*gridDimPME.y/box.y;
    fi.z*=l*q*gridDimPME.z/box.z;
    fi.x+=__shfl_down_sync(0xFFFFFFFF,fi.x,1);
    fi.x+=__shfl_down_sync(0xFFFFFFFF,fi.x,2);
    fi.y+=__shfl_down_sync(0xFFFFFFFF,fi.y,1);
    fi.y+=__shfl_down_sync(0xFFFFFFFF,fi.y,2);
    fi.z+=__shfl_down_sync(0xFFFFFFFF,fi.z,1);
    fi.z+=__shfl_down_sync(0xFFFFFFFF,fi.z,2);
    if ((i&3)==0) {
      at_real3_inc(&force[i/order], fi);
    }
  }

  // Energy, if requested
  if (energy) {
    lEnergy*=l*q;
    real_sum_reduce(lEnergy,sEnergy,energy);
  }
}
// NYI - not sure if I need this comment for a template
// getforce_angle_kernel<<<(N+BLBO-1)/BLBO,BLBO,shMem,p->bondedStream>>>(N,p->angles_d,(real3*)s->position_d,(real3*)s->force_d,s->orthBox,m->lambda_d,m->lambdaForce_d,pEnergy);
// __global__ void getforce_angle_kernel(int angleCount,struct AnglePotential *angles,real3 *position,real3 *force,real3 box,real *lambda,real *lambdaForce,real *energy)

void getforce_ewald(System *system,bool calcEnergy)
{
  Potential *p=system->potential;
  State *s=system->state;
  Msld *m=system->msld;
  Run *r=system->run;
  int N=p->atomCount;
  int shMem=0;
  real *pEnergy=NULL;

  if (calcEnergy) {
    shMem=BLNB*sizeof(real)/32;
    pEnergy=s->energy_d+eenbrecip;
  }

  cudaMemsetAsync(p->chargeGridPME_d,0,p->gridDimPME[0]*p->gridDimPME[1]*p->gridDimPME[2]*sizeof(cufftReal),r->nbrecipStream);

  // Spread kernel
  getforce_ewald_spread_kernel<<<(4*N+BLNB-1)/BLNB,BLNB,0,r->nbrecipStream>>>(N,p->charge_d,m->atomBlock_d,(real3*)s->position_d,s->orthBox,s->lambda_d,((int3*)p->gridDimPME)[0],p->chargeGridPME_d);

  cufftExecR2C(p->planFFTPME,p->chargeGridPME_d,p->fourierGridPME_d);

  // Convolution kernel
  dim3 blockCount((p->gridDimPME[0]+8-1)/8,(p->gridDimPME[1]+8-1)/8,(p->gridDimPME[2]/2+1+8-1)/8);
  dim3 blockSize(8,8,8);
  getforce_ewald_convolution_kernel<<<blockCount,blockSize,0,r->nbrecipStream>>>(((int3*)p->gridDimPME)[0],p->fourierGridPME_d,p->bGridPME_d,system->run->betaEwald,s->orthBox);
/*
  dim3 blockCount2((p->gridDimPME[0]+16-1)/16,(p->gridDimPME[1]/2+1+16-1)/16,1);
  dim3 blockSize2(16,16,2-(p->gridDimPME[2]&1)); // if third grid dim is even, we need two z threads, one for 0 plane and one for mid plane in k space
  getforce_ewald_regularize_kernel<<<blockCount2,blockSize2,0,p->nbrecipStream>>>(((int3*)p->gridDimPME)[0],p->fourierGridPME_d);
*/

  cufftExecC2R(p->planIFFTPME,p->fourierGridPME_d,p->potentialGridPME_d);

  // Gather kernel
  getforce_ewald_gather_kernel<<<(4*N+BLNB-1)/BLNB,BLNB,shMem,r->nbrecipStream>>>(N,p->charge_d,m->atomBlock_d,((int3*)p->gridDimPME)[0],p->potentialGridPME_d,(real3*)s->position_d,(real3*)s->force_d,s->orthBox,s->lambda_d,s->lambdaForce_d,pEnergy);
}
