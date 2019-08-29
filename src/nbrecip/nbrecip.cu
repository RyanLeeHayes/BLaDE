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
      atomicAdd(&lambdaForce[b],2*l*lEnergy);
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

  if (r->calcTermFlag[eenbrecipself]==false) return;

  if (calcEnergy) {
    shMem=BLNB*sizeof(real)/32;
    pEnergy=s->energy_d+eenbrecipself;
  }

  getforce_ewaldself_kernel<<<(N+BLNB-1)/BLNB,BLNB,shMem,r->nbrecipStream>>>(N,p->charge_d,prefactor,m->atomBlock_d,s->lambda_d,s->lambdaForce_d,pEnergy);
}



// getforce_ewald_spread_kernel<<<>>>(N,p->charge_d,m->atomBlock_d,(real3*)s->position_d,s->orthBox,m->lambda_d,((int3*)p->gridDimPME)[0],p->chargeGridPME_d);
template <int order>
__global__ void getforce_ewald_spread_kernel(int atomCount,real *charge,int *atomBlock,real3* position,real3 box,real *lambda,int3 gridDimPME,real *chargeGridPME)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real q;
  int b;
  real l=1;
  real3 xi; // position
  real u; // fractional coordinate remainder
  int3 u0; // index of grid point
  real Meven,Modd; // even and odd order B splines
  real3 density;
  // real3 dDensity;
  real3 dIndex;
  int j,jx,jy,jz,jxs,jys,jzs;
  int3 index;
  int iThread=rectify_modulus(threadIdx.x,32); // threadIdx.x&31; // within warp
  int iAtom=i/8;
  int threadOfAtom=rectify_modulus(i,8); // iThread%order;

  u0=make_int3(0,0,0);

  if (iAtom<atomCount) {
    q=charge[iAtom];

    // Scaling
    b=atomBlock[iAtom];
    if (b) {
      l=lambda[b];
    }
    q*=l;

    xi=position[iAtom];

    // Get grid position
    u=xi.x*gridDimPME.x/box.x;
    u0.x=(int)floor(u);
    u-=u0.x;
    u0.x=nearby_modulus(u0.x,gridDimPME.x);
    u+=(order-1-threadOfAtom); // very important 
  } else {
    u=0;
  }

  Modd=0;
  Meven=(threadOfAtom==(order-1))?u:0;
  Meven=(threadOfAtom==(order-2))?2-u:Meven; // Second order B spline
  for (j=2; j<order; j+=2) {
    Modd=u*Meven+(j+1-u)*__shfl_sync(0xFFFFFFFF,Meven,iThread+1);
    Modd*=1.0f/j; // 1/(n-1)
    Meven=u*Modd+(j+2-u)*__shfl_sync(0xFFFFFFFF,Modd,iThread+1);
    Meven*=1.0f/(j+1); // 1/(n-1)
  }
  density.x=Meven;
  // dDensity.x=Modd-__shfl_sync(0xFFFFFFFF,Modd,iThread+1);

  if (iAtom<atomCount) {
    u=xi.y*gridDimPME.y/box.y;
    u0.y=(int)floor(u);
    u-=u0.y;
    u0.y=nearby_modulus(u0.y,gridDimPME.y);
    u+=(order-1-threadOfAtom); // very important 
  } else {
    u=0;
  }

  Modd=0;
  Meven=(threadOfAtom==(order-1))?u:0;
  Meven=(threadOfAtom==(order-2))?2-u:Meven; // Second order B spline
  for (j=2; j<order; j+=2) {
    Modd=u*Meven+(j+1-u)*__shfl_sync(0xFFFFFFFF,Meven,iThread+1);
    Modd*=1.0f/j; // 1/(n-1)
    Meven=u*Modd+(j+2-u)*__shfl_sync(0xFFFFFFFF,Modd,iThread+1);
    Meven*=1.0f/(j+1); // 1/(n-1)
  }
  density.y=Meven;
  // dDensity.y=Modd-__shfl_sync(0xFFFFFFFF,Modd,iThread+1);

  if (iAtom<atomCount) {
    u=xi.z*gridDimPME.z/box.z;
    u0.z=(int)floor(u);
    u-=u0.z;
    u0.z=nearby_modulus(u0.z,gridDimPME.z);
    u+=(order-1-threadOfAtom); // very important 
  } else {
    u=0;
  }

  Modd=0;
  Meven=(threadOfAtom==(order-1))?u:0;
  Meven=(threadOfAtom==(order-2))?2-u:Meven; // Second order B spline
  for (j=2; j<order; j+=2) {
    Modd=u*Meven+(j+1-u)*__shfl_sync(0xFFFFFFFF,Meven,iThread+1);
    Modd*=1.0f/j; // 1/(n-1)
    Meven=u*Modd+(j+2-u)*__shfl_sync(0xFFFFFFFF,Modd,iThread+1);
    Meven*=1.0f/(j+1); // 1/(n-1)
  }
  density.z=Meven;
  // dDensity.z=Modd-__shfl_sync(0xFFFFFFFF,Modd,iThread+1);

  jxs=(threadIdx.x&4)==4;
  jys=(threadIdx.x&2)==2;
  jzs=(threadIdx.x&1);
  for (jx=0; jx<order/2; jx++) {
    dIndex.x=__shfl_sync(0xFFFFFFFF,density.x,threadIdx.x-threadOfAtom+2*jx+jxs);
    index.x=over_modulus(u0.x+2*jx+jxs,gridDimPME.x);
    for (jy=0; jy<order/2; jy++) {
      dIndex.y=__shfl_sync(0xFFFFFFFF,density.y,threadIdx.x-threadOfAtom+2*jy+jys);
      index.y=index.x*gridDimPME.y+over_modulus(u0.y+2*jy+jys,gridDimPME.y);
      for (jz=0; jz<order/2; jz++) {
        dIndex.z=__shfl_sync(0xFFFFFFFF,density.z,threadIdx.x-threadOfAtom+2*jz+jzs);
        index.z=index.y*gridDimPME.z+over_modulus(u0.z+2*jz+jzs,gridDimPME.z);
        if (iAtom<atomCount) {
          atomicAdd(&chargeGridPME[index.z],q*dIndex.x*dIndex.y*dIndex.z);
        }
      }
    }
  }
}

// getforce_ewald_convolution_kernel<<<blockCount,blockSize,0,p->nbrecipStream>>>(((int3*)gridDimPME)[0],p->fourierGridPME_d,p->bGridPME_d,system->run->betaEwald,s->orthoBox)
__global__ void getforce_ewald_convolution_kernel(int3 gridDimPME,myCufftComplex *fourierGridPME,real *bGridPME,real betaEwald,real3 box)
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
    factor*=(((real)0.5)*kELECTRIC/((real)M_PI));
    factor*=exp(-((real)M_PI)*((real)M_PI)*k2/(betaEwald*betaEwald));
    factor/=V*k2;
    factor=(ijk==0?0:factor);
    fourierGridPME[ijk].x*=factor;
    fourierGridPME[ijk].y*=factor;
  }
}

// getforce_ewald_gather_kernel<<<>>>(N,p->charge_d,prefactor,m->atomBlock_d,((int3*)p->gridDimPME)[0],p->potentialGridPME_d,(real3*)s->position_d,(real3*)s->force_d,s->orthBox,m->lambda_d,m->lambdaForce_d,pEnergy);
template <int order>
__global__ void getforce_ewald_gather_kernel(
  int atomCount,
  real *charge,
  int *atomBlock,
  int3 gridDimPME,
#ifdef USE_TEXTURE
  cudaTextureObject_t potentialGridPME,
#else
  real *potentialGridPME,
#endif
  real3 *position,
  real3 *force,
  real3 box,
  real *lambda,
  real *lambdaForce,
  real *energy)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real q;
  int b;
  real l=1;
  real3 xi; // position
  real3 fi;
  real u; // fractional coordinate remainder
  int3 u0; // index of grid point
  real Meven,Modd; // even and odd order B splines
  real3 density;
  real3 dDensity;
  real3 dIndex, dDIndex;
  int j,jx,jy,jz,jxs,jys,jzs;
  int3 index;
  int iThread=rectify_modulus(threadIdx.x,32); // threadIdx.x&31; // within warp
  int iAtom=i/8;
  int threadOfAtom=rectify_modulus(i,8); // iThread%order;
  real passValue;
  real lEnergy=0;
  extern __shared__ real sEnergy[];

  u0=make_int3(0,0,0);

  if (iAtom<atomCount) {
    q=charge[iAtom];

    // Scaling
    b=atomBlock[iAtom];
    if (b) {
      l=lambda[b];
    }
    // q*=l; // do this scaling later in the kernel

    xi=position[iAtom];

    // Get grid position
    u=xi.x*gridDimPME.x/box.x;
    u0.x=(int)floor(u);
    u-=u0.x;
    u0.x=nearby_modulus(u0.x,gridDimPME.x);
    u+=(order-1-threadOfAtom); // very important 
  } else {
    u=0;
  }

  Modd=0;
  Meven=(threadOfAtom==(order-1))?u:0;
  Meven=(threadOfAtom==(order-2))?2-u:Meven; // Second order B spline
  for (j=2; j<order; j+=2) {
    Modd=u*Meven+(j+1-u)*__shfl_sync(0xFFFFFFFF,Meven,iThread+1);
    Modd*=1.0f/j; // 1/(n-1)
    Meven=u*Modd+(j+2-u)*__shfl_sync(0xFFFFFFFF,Modd,iThread+1);
    Meven*=1.0f/(j+1); // 1/(n-1)
  }
  density.x=Meven;
  dDensity.x=Modd-__shfl_sync(0xFFFFFFFF,Modd,iThread+1);

  if (iAtom<atomCount) {
    u=xi.y*gridDimPME.y/box.y;
    u0.y=(int)floor(u);
    u-=u0.y;
    u0.y=nearby_modulus(u0.y,gridDimPME.y);
    u+=(order-1-threadOfAtom); // very important 
  } else {
    u=0;
  }

  Modd=0;
  Meven=(threadOfAtom==(order-1))?u:0;
  Meven=(threadOfAtom==(order-2))?2-u:Meven; // Second order B spline
  for (j=2; j<order; j+=2) {
    Modd=u*Meven+(j+1-u)*__shfl_sync(0xFFFFFFFF,Meven,iThread+1);
    Modd*=1.0f/j; // 1/(n-1)
    Meven=u*Modd+(j+2-u)*__shfl_sync(0xFFFFFFFF,Modd,iThread+1);
    Meven*=1.0f/(j+1); // 1/(n-1)
  }
  density.y=Meven;
  dDensity.y=Modd-__shfl_sync(0xFFFFFFFF,Modd,iThread+1);

  if (iAtom<atomCount) {
    u=xi.z*gridDimPME.z/box.z;
    u0.z=(int)floor(u);
    u-=u0.z;
    u0.z=nearby_modulus(u0.z,gridDimPME.z);
    u+=(order-1-threadOfAtom); // very important 
  } else {
    u=0;
  }

  Modd=0;
  Meven=(threadOfAtom==(order-1))?u:0;
  Meven=(threadOfAtom==(order-2))?2-u:Meven; // Second order B spline
  for (j=2; j<order; j+=2) {
    Modd=u*Meven+(j+1-u)*__shfl_sync(0xFFFFFFFF,Meven,iThread+1);
    Modd*=1.0f/j; // 1/(n-1)
    Meven=u*Modd+(j+2-u)*__shfl_sync(0xFFFFFFFF,Modd,iThread+1);
    Meven*=1.0f/(j+1); // 1/(n-1)
  }
  density.z=Meven;
  dDensity.z=Modd-__shfl_sync(0xFFFFFFFF,Modd,iThread+1);

  // Everything is the same as the spread kernel up to this point
  jxs=(threadIdx.x&4)==4;
  jys=(threadIdx.x&2)==2;
  jzs=(threadIdx.x&1);
  fi.x=0;
  fi.y=0;
  fi.z=0;
  for (jx=0; jx<order/2; jx++) {
    dIndex.x=__shfl_sync(0xFFFFFFFF,density.x,threadIdx.x-threadOfAtom+2*jx+jxs);
    dDIndex.x=__shfl_sync(0xFFFFFFFF,dDensity.x,threadIdx.x-threadOfAtom+2*jx+jxs);
    index.x=over_modulus(u0.x+2*jx+jxs,gridDimPME.x);
    for (jy=0; jy<order/2; jy++) {
      dIndex.y=__shfl_sync(0xFFFFFFFF,density.y,threadIdx.x-threadOfAtom+2*jy+jys);
      dDIndex.y=__shfl_sync(0xFFFFFFFF,dDensity.y,threadIdx.x-threadOfAtom+2*jy+jys);
      index.y=index.x*gridDimPME.y+over_modulus(u0.y+2*jy+jys,gridDimPME.y);
      for (jz=0; jz<order/2; jz++) {
        dIndex.z=__shfl_sync(0xFFFFFFFF,density.z,threadIdx.x-threadOfAtom+2*jz+jzs);
        dDIndex.z=__shfl_sync(0xFFFFFFFF,dDensity.z,threadIdx.x-threadOfAtom+2*jz+jzs);
        index.z=index.y*gridDimPME.z+over_modulus(u0.z+2*jz+jzs,gridDimPME.z);
        if (iAtom<atomCount) {
#ifdef USE_TEXTURE
          real P=tex1Dfetch<real>(potentialGridPME,index.z);
#else
          real P=potentialGridPME[index.z];
#endif
          fi.x+=P*dDIndex.x*dIndex.y*dIndex.z;
          fi.y+=P*dIndex.x*dDIndex.y*dIndex.z;
          fi.z+=P*dIndex.x*dIndex.y*dDIndex.z;
          if (b || energy) {
            lEnergy+=P*dIndex.x*dIndex.y*dIndex.z;
          }
        }
      }
    }
  }

  // Reductions
  fi.x+=__shfl_down_sync(0xFFFFFFFF,fi.x,1);
  fi.y+=__shfl_down_sync(0xFFFFFFFF,fi.y,1);
  fi.z+=__shfl_down_sync(0xFFFFFFFF,fi.z,1);
  lEnergy+=__shfl_down_sync(0xFFFFFFFF,lEnergy,1);
  fi.x+=__shfl_down_sync(0xFFFFFFFF,fi.x,2);
  fi.y+=__shfl_down_sync(0xFFFFFFFF,fi.y,2);
  fi.z+=__shfl_down_sync(0xFFFFFFFF,fi.z,2);
  lEnergy+=__shfl_down_sync(0xFFFFFFFF,lEnergy,2);
  fi.x+=__shfl_down_sync(0xFFFFFFFF,fi.x,4);
  fi.y+=__shfl_down_sync(0xFFFFFFFF,fi.y,4);
  fi.z+=__shfl_down_sync(0xFFFFFFFF,fi.z,4);
  lEnergy+=__shfl_down_sync(0xFFFFFFFF,lEnergy,4);
  // Reduction only correct for threadOfAtom==0

  // Lambda force
  if (iAtom<atomCount) {
    if (b && threadOfAtom==0) {
      atomicAdd(&lambdaForce[b],2*q*lEnergy);
    }
  }

  // Spatial force
  if (iAtom<atomCount) {
    fi.x*=2*l*q*gridDimPME.x/box.x;
    fi.y*=2*l*q*gridDimPME.y/box.y;
    fi.z*=2*l*q*gridDimPME.z/box.z;
    if (threadOfAtom==0) {
      at_real3_inc(&force[iAtom], fi);
    }
  }

  // Energy, if requested
  if (energy) {
    if (threadOfAtom==0) {
      lEnergy*=l*q;
    } else {
      lEnergy=0;
    }
#warning "Using reduction without shared memory"
    // real_sum_reduce(lEnergy,sEnergy,energy);
    real_sum_reduce(lEnergy,energy);
  }
}

void getforce_ewald(System *system,bool calcEnergy)
{
  Potential *p=system->potential;
  State *s=system->state;
  Msld *m=system->msld;
  Run *r=system->run;
  int N=p->atomCount;
  int shMem=0;
  real *pEnergy=NULL;

  if (r->calcTermFlag[eenbrecip]==false) return;

  if (calcEnergy) {
    shMem=BLNB*sizeof(real)/32;
    pEnergy=s->energy_d+eenbrecip;
  }

  cudaMemsetAsync(p->chargeGridPME_d,0,p->gridDimPME[0]*p->gridDimPME[1]*p->gridDimPME[2]*sizeof(myCufftReal),r->nbrecipStream);

  // Setup for spread and gather
  int order=r->orderEwald;
  int spreadGatherBlocks=(N + BLNB/8 - 1)/(BLNB/8);

  // Spread kernel
  // getforce_ewald_spread_kernel<<<(4*N+BLNB-1)/BLNB,BLNB,0,r->nbrecipStream>>>(N,p->charge_d,m->atomBlock_d,(real3*)s->position_d,s->orthBox,s->lambda_d,((int3*)p->gridDimPME)[0],p->chargeGridPME_d);
  if (order==4) {
    getforce_ewald_spread_kernel<4><<<spreadGatherBlocks,BLNB,0,r->nbrecipStream>>>(N,p->charge_d,m->atomBlock_d,(real3*)s->position_d,s->orthBox,s->lambda_d,((int3*)p->gridDimPME)[0],p->chargeGridPME_d);
  } else if (order==6) {
    getforce_ewald_spread_kernel<6><<<spreadGatherBlocks,BLNB,0,r->nbrecipStream>>>(N,p->charge_d,m->atomBlock_d,(real3*)s->position_d,s->orthBox,s->lambda_d,((int3*)p->gridDimPME)[0],p->chargeGridPME_d);
  } else if (order==8) {
    getforce_ewald_spread_kernel<8><<<spreadGatherBlocks,BLNB,0,r->nbrecipStream>>>(N,p->charge_d,m->atomBlock_d,(real3*)s->position_d,s->orthBox,s->lambda_d,((int3*)p->gridDimPME)[0],p->chargeGridPME_d);
  }

  myCufftExecR2C(p->planFFTPME,p->chargeGridPME_d,p->fourierGridPME_d);

  // Convolution kernel
  dim3 blockCount((p->gridDimPME[0]+8-1)/8,(p->gridDimPME[1]+8-1)/8,(p->gridDimPME[2]/2+1+8-1)/8);
  dim3 blockSize(8,8,8);
  getforce_ewald_convolution_kernel<<<blockCount,blockSize,0,r->nbrecipStream>>>(((int3*)p->gridDimPME)[0],p->fourierGridPME_d,p->bGridPME_d,system->run->betaEwald,s->orthBox);
/*
  dim3 blockCount2((p->gridDimPME[0]+16-1)/16,(p->gridDimPME[1]/2+1+16-1)/16,1);
  dim3 blockSize2(16,16,2-(p->gridDimPME[2]&1)); // if third grid dim is even, we need two z threads, one for 0 plane and one for mid plane in k space
  getforce_ewald_regularize_kernel<<<blockCount2,blockSize2,0,p->nbrecipStream>>>(((int3*)p->gridDimPME)[0],p->fourierGridPME_d);
*/

  myCufftExecC2R(p->planIFFTPME,p->fourierGridPME_d,p->potentialGridPME_d);

  // Gather kernel
  // getforce_ewald_gather_kernel<<<(4*N+BLNB-1)/BLNB,BLNB,shMem,r->nbrecipStream>>>(N,p->charge_d,m->atomBlock_d,((int3*)p->gridDimPME)[0],p->potentialGridPME_d,(real3*)s->position_d,(real3*)s->force_d,s->orthBox,s->lambda_d,s->lambdaForce_d,pEnergy);
  if (order==4) {
    getforce_ewald_gather_kernel<4><<<spreadGatherBlocks,BLNB,shMem,r->nbrecipStream>>>(N,p->charge_d,m->atomBlock_d,((int3*)p->gridDimPME)[0],
#ifdef USE_TEXTURE
      p->potentialGridPME_tex,
#else
      p->potentialGridPME_d,
#endif
      (real3*)s->position_d,(real3*)s->force_d,s->orthBox,s->lambda_d,s->lambdaForce_d,pEnergy);
  } else if (order==6) {
    getforce_ewald_gather_kernel<6><<<spreadGatherBlocks,BLNB,shMem,r->nbrecipStream>>>(N,p->charge_d,m->atomBlock_d,((int3*)p->gridDimPME)[0],
#ifdef USE_TEXTURE
      p->potentialGridPME_tex,
#else
      p->potentialGridPME_d,
#endif
      (real3*)s->position_d,(real3*)s->force_d,s->orthBox,s->lambda_d,s->lambdaForce_d,pEnergy);
  } else if (order==8) {
    getforce_ewald_gather_kernel<8><<<spreadGatherBlocks,BLNB,shMem,r->nbrecipStream>>>(N,p->charge_d,m->atomBlock_d,((int3*)p->gridDimPME)[0],
#ifdef USE_TEXTURE
      p->potentialGridPME_tex,
#else
      p->potentialGridPME_d,
#endif
      (real3*)s->position_d,(real3*)s->force_d,s->orthBox,s->lambda_d,s->lambdaForce_d,pEnergy);
  }
}
