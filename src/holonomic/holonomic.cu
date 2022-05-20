#include "holonomic/holonomic.h"
#include "system/system.h"
#include "run/run.h"
#include "system/potential.h"
#include "system/state.h"

#include "main/real3.h"

#define MAXITERATION 10



template <bool flagBox,typename box_type>
__global__ void holonomic_velocity_triangle_kernel(int N,struct TriangleCons *cons,struct LeapState ls,box_type box)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real_x AT[3][3]; // A transpose
  real_x b[3];
  real_x lambda[3];
  // A*lambda=b, solve for lambda
  real_x im[3];
  real3_x x[3];
  real3_v v[3];
  // real b02[3];
  real3_x dx[3];
  int j,k,l,m,n,nSign;

  if (i<N) {
    for (j=0; j<3; j++) {
      x[j]=((real3_x*)ls.x)[cons[i].idx[j]];
      v[j]=((real3_v*)ls.v)[cons[i].idx[j]];
      im[j]=ls.ism[3*cons[i].idx[j]];
      im[j]*=im[j];
      // b02[j]=cons[i].b0[j]*cons[i].b0[j];
    }
    for (j=0; j<3; j++) {
      for (k=j+1; k<3; k++) {
        l=j+k-1; // difference index 01->0 02->1 12->2
        dx[l]=real3_subpbc<flagBox>(x[j],x[k],box);
      }
    }
    for (j=0; j<3; j++) {
      for (k=j+1; k<3; k++) {
        l=j+k-1; // difference index 01->0 02->1 12->2
        m=3-(j+k); // not j or k
        b[l]=-real3_dot<real_x>(real3_sub(v[j],v[k]),dx[l]);
        // AT[l][l]=(im[j]+im[k])*b02[l];
        AT[l][l]=(im[j]+im[k])*real3_mag2<real_x>(dx[l]);
        n=j+m-1; // difference index for j and m
        nSign=(j<m?1:-1);
        // AT[n][l]=0.5*im[j]*(b02[l]+b02[n]-b02[3-(l+n)]);
        AT[n][l]=im[j]*nSign*real3_dot<real_x>(dx[l],dx[n]);
        n=k+m-1; // difference index for k and m
        nSign=(k<m?-1:1);
        // AT[n][l]=0.5*im[k]*(b02[l]+b02[n]-b02[3-(l+n)]);
        AT[n][l]=im[k]*nSign*real3_dot<real_x>(dx[l],dx[n]);
      }
    }
    for (j=0; j<3; j++) {
      for (k=j+1; k<3; k++) {
        l=j+k-1; // difference index 01->0 02->1 12->2
        m=l+1; // l+1
        m=(m==3?0:m);
        n=3-(l+m); // l+2
        real3_x partialDet=real3_cross(((real3_x*)AT)[m],((real3_x*)AT)[n]);
        lambda[l]=real3_dot<real_x>(((real3_x*)b)[0],partialDet)/real3_dot<real_x>(((real3_x*)AT)[l],partialDet);
      }
    }
    for (j=0; j<3; j++) {
      for (k=j+1; k<3; k++) {
        l=j+k-1; // difference index 01->0 02->1 12->2
        real3_scaleinc(&v[j], lambda[l]*im[j],dx[l]);
        real3_scaleinc(&v[k],-lambda[l]*im[k],dx[l]);
      }
    }
    for (j=0; j<3; j++) {
      ((real3_v*)ls.v)[cons[i].idx[j]]=v[j];
    }
  }
}

template <bool flagBox,typename box_type>
__global__ void holonomic_velocity_branch1_kernel(int N,struct Branch1Cons *cons,struct LeapState ls,box_type box)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real_x AT; // A transpose
  real_x b;
  real_x lambda;
  // A*lambda=b, solve for lambda
  real_x im[2];
  real3_x x[2];
  real3_v v[2];
  real3_x dx;
  int j;

  if (i<N) {
    for (j=0; j<2; j++) {
      x[j]=((real3_x*)ls.x)[cons[i].idx[j]];
      v[j]=((real3_v*)ls.v)[cons[i].idx[j]];
      im[j]=ls.ism[3*cons[i].idx[j]];
      im[j]*=im[j];
    }
    dx=real3_subpbc<flagBox>(x[0],x[1],box);
    b=-real3_dot<real_x>(real3_sub(v[0],v[1]),dx);
    AT=(im[0]+im[1])*real3_dot<real_x>(dx,dx);
    lambda=b/AT;
    real3_scaleinc(&v[0], lambda*im[0],dx);
    real3_scaleinc(&v[1],-lambda*im[1],dx);
    for (j=0; j<2; j++) {
      ((real3_v*)ls.v)[cons[i].idx[j]]=v[j];
    }
  }
}

template <bool flagBox,typename box_type>
__global__ void holonomic_velocity_branch2_kernel(int N,struct Branch2Cons *cons,struct LeapState ls,box_type box)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real_x AT[2][2]; // A transpose
  real_x b[2];
  real_x lambda[2];
  // A*lambda=b, solve for lambda
  real_x im[3];
  real3_x x[3];
  real3_v v[3];
  real3_x dx[2];
  int j,jp;

  if (i<N) {
    for (j=0; j<3; j++) {
      x[j]=((real3_x*)ls.x)[cons[i].idx[j]];
      v[j]=((real3_v*)ls.v)[cons[i].idx[j]];
      im[j]=ls.ism[3*cons[i].idx[j]];
      im[j]*=im[j];
    }
    for (j=0; j<2; j++) {
      dx[j]=real3_subpbc<flagBox>(x[0],x[j+1],box);
    }
    for (j=0; j<2; j++) {
      jp=1-j;
      b[j]=-real3_dot<real_x>(real3_sub(v[0],v[j+1]),dx[j]);
      AT[j][j]=(im[0]+im[j+1])*real3_dot<real_x>(dx[j],dx[j]);
      AT[jp][j]=im[0]*real3_dot<real_x>(dx[jp],dx[j]);
    }
    real_x invDet=1/(AT[0][0]*AT[1][1]-AT[0][1]*AT[1][0]);
    for (j=0; j<2; j++) {
      jp=1-j;
      lambda[j]=(b[j]*AT[jp][jp]-b[jp]*AT[jp][j])*invDet;
    }
    for (j=0; j<2; j++) {
      real3_scaleinc(&v[0],   lambda[j]*im[0],  dx[j]);
      real3_scaleinc(&v[j+1],-lambda[j]*im[j+1],dx[j]);
    }
    for (j=0; j<3; j++) {
      ((real3_v*)ls.v)[cons[i].idx[j]]=v[j];
    }
  }
}

template <bool flagBox,typename box_type>
__global__ void holonomic_velocity_branch3_kernel(int N,struct Branch3Cons *cons,struct LeapState ls,box_type box)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real_x AT[3][3]; // A transpose
  real_x b[3];
  real_x lambda[3];
  // A*lambda=b, solve for lambda
  real_x im[4];
  real3_x x[4];
  real3_v v[4];
  real3_x dx[3];
  int j,jp,jpp;

  if (i<N) {
    for (j=0; j<4; j++) {
      x[j]=((real3_x*)ls.x)[cons[i].idx[j]];
      v[j]=((real3_v*)ls.v)[cons[i].idx[j]];
      im[j]=ls.ism[3*cons[i].idx[j]];
      im[j]*=im[j];
    }
    for (j=0; j<3; j++) {
      dx[j]=real3_subpbc<flagBox>(x[0],x[j+1],box);
    }
    for (j=0; j<3; j++) {
      // jp=(j+1)%3;
      // jpp=(j+2)%3;
        jp=((j==2)?0:(j+1));
        jpp=3-(j+jp);
      b[j]=-real3_dot<real_x>(real3_sub(v[0],v[j+1]),dx[j]);
      AT[j][j]=(im[0]+im[j+1])*real3_dot<real_x>(dx[j],dx[j]);
      AT[jp][j]=im[0]*real3_dot<real_x>(dx[jp],dx[j]);
      AT[jpp][j]=im[0]*real3_dot<real_x>(dx[jpp],dx[j]);
    }
    for (j=0; j<3; j++) {
      // jp=(j+1)%3;
      // jpp=(j+2)%3;
        jp=((j==2)?0:(j+1));
        jpp=3-(j+jp);
      real3_x partialDet=real3_cross(((real3_x*)AT)[jp],((real3_x*)AT)[jpp]);
      lambda[j]=real3_dot<real_x>(((real3_x*)b)[0],partialDet)/real3_dot<real_x>(((real3_x*)AT)[j],partialDet);
    }
    for (j=0; j<3; j++) {
      real3_scaleinc(&v[0],   lambda[j]*im[0],  dx[j]);
      real3_scaleinc(&v[j+1],-lambda[j]*im[j+1],dx[j]);
    }
    for (j=0; j<4; j++) {
      ((real3_v*)ls.v)[cons[i].idx[j]]=v[j];
    }
  }
}

template <bool flagBox,typename box_type>
void holonomic_velocityT(System *system,box_type box)
{
  Run *r=system->run;
  State *s=system->state;
  Potential *p=system->potential;
  int N;

  N=p->triangleConsCount;
  if (N) holonomic_velocity_triangle_kernel<flagBox><<<(N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(N,p->triangleCons_d,s->leapState[0],box);

  N=p->branch1ConsCount;
  if (N) holonomic_velocity_branch1_kernel<flagBox><<<(N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(N,p->branch1Cons_d,s->leapState[0],box);

  N=p->branch2ConsCount;
  if (N) holonomic_velocity_branch2_kernel<flagBox><<<(N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(N,p->branch2Cons_d,s->leapState[0],box);

  N=p->branch3ConsCount;
  if (N) holonomic_velocity_branch3_kernel<flagBox><<<(N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(N,p->branch3Cons_d,s->leapState[0],box);
}

void holonomic_velocity(System *system)
{
  if (system->state->typeBox) {
    holonomic_velocityT<true>(system,system->state->tricBox);
  } else {
    holonomic_velocityT<false>(system,system->state->orthBox);
  }
}



// __global__ void holonomic_position_triangle_kernel(int N,struct TriangleCons *cons,struct LeapState ls,struct LeapParms2 lp,real_x *xPrevious,real3_x box)
// {}
//   int i=blockIdx.x*blockDim.x+threadIdx.x;
template <bool flagBox,typename box_type>
__device__ inline void holonomic_position_triangle_kernel(int N,int B0,struct TriangleCons *cons,struct LeapState ls,struct LeapParms2 lp,real_x *xPrevious,box_type box,real tolerance)
{
  int i=(blockIdx.x-B0)*blockDim.x+threadIdx.x;
  real_x invMass[3];
  real_x mass[3];
  real3_x xPrev[3];
  real3_x x[3];
  real3_x xNew[3];
  real3_x xShift[3];
  real b02[3];
  real3_x comPrev;
  real3_x com; // center of mass
  real3_x intCoord[3]; // yHat along xPrev[0]-comPrev, zHat along (xPrev[1]-xPrev[0]) cross (xPrev[2]-xPrev[0])
  real3_x R[3]; // Rotation matrix
  int j;

  if (i<N) {
    comPrev=real3_reset<real3_x>();;
    com=real3_reset<real3_x>();
    for (j=0; j<3; j++) {
      xPrev[j]=((real3_x*)xPrevious)[cons[i].idx[j]];
      if (j>0) {
        xShift[j]=real3_sub(real3_subpbc<flagBox>(xPrev[j],xPrev[0],box),real3_sub(xPrev[j],xPrev[0]));
      } else {
        xShift[0]=real3_reset<real3_x>();
      }
      xPrev[j]=real3_add(xPrev[j],xShift[j]);
      x[j]=real3_add(((real3_x*)ls.x)[cons[i].idx[j]],xShift[j]);
      invMass[j]=ls.ism[3*cons[i].idx[j]];
      invMass[j]*=invMass[j];
      mass[j]=1/invMass[j];
      b02[j]=cons[i].b0[j];
      b02[j]*=b02[j];
      real3_scaleinc(&comPrev,mass[j],xPrev[j]);
      real3_scaleinc(&com,mass[j],x[j]);
    }
    real_x invM=1/(mass[0]+mass[1]+mass[2]);
    real3_scaleself(&comPrev,invM);
    real3_scaleself(&com,invM);

    // Move COM to origin
    for (j=0; j<3; j++) {
      xPrev[j]=real3_sub(xPrev[j],comPrev);
      x[j]=real3_sub(x[j],com);
    }

    // Define internal coordinates
    intCoord[1]=xPrev[0];
    real3_scaleself(&intCoord[1],1/real3_mag<real_x>(intCoord[1]));
    intCoord[2]=real3_cross(xPrev[1],xPrev[2]);
    real3_scaleself(&intCoord[2],1/real3_mag<real_x>(intCoord[2]));
    intCoord[0]=real3_cross(intCoord[1],intCoord[2]);

    // Rectify starting positions
    real_x r0=invM*sqrt(mass[1]*mass[1]*b02[0]+mass[1]*mass[2]*(b02[0]+b02[1]-b02[2])+mass[2]*mass[2]*b02[1]);
    real_x r1=invM*sqrt(mass[0]*mass[0]*b02[0]+mass[0]*mass[2]*(b02[0]+b02[2]-b02[1])+mass[2]*mass[2]*b02[2]);
    real_x r2=invM*sqrt(mass[0]*mass[0]*b02[1]+mass[0]*mass[1]*(b02[1]+b02[2]-b02[0])+mass[1]*mass[1]*b02[2]);
    xPrev[0]=real3_scale<real3_x>(r0,intCoord[1]);
    real_x r1proj=(r0*r0+r1*r1-b02[0])/(2*r0);
    real_x r2proj=(r0*r0+r2*r2-b02[1])/(2*r0);
    xPrev[1]=real3_scale<real3_x>(r1proj,intCoord[1]);
    xPrev[2]=real3_scale<real3_x>(r2proj,intCoord[1]);
    real3_scaleinc(&xPrev[1],-sqrt(r1*r1-r1proj*r1proj),intCoord[0]);
    real3_scaleinc(&xPrev[2],sqrt(r2*r2-r2proj*r2proj),intCoord[0]);

    // Find psi and phi
    real_x Z_A1=real3_dot<real_x>(intCoord[2],x[0]);
    real_x r_a=real3_mag<real_x>(xPrev[0]);
    real_x sinPhi=Z_A1/r_a;
    real_x cosPhi=sqrt(1-sinPhi*sinPhi);
    real_x Z_B1=real3_dot<real_x>(intCoord[2],x[1]);
    real_x r_bX=-real3_dot<real_x>(intCoord[0],xPrev[1]);
    real_x r_bY=-real3_dot<real_x>(intCoord[1],xPrev[1]);
    real_x sinPsi=(Z_B1+r_bY*sinPhi)/(r_bX*cosPhi);
    real_x cosPsi=sqrt(1-sinPsi*sinPsi);

    // Rotate by psi around Y=intCoord[1]
    real3_rotation_matrix(R,intCoord[1],cosPsi,sinPsi);
    for (j=0; j<3; j++) {
      xNew[j]=real3_rotate<real3_x,real_x>(R,xPrev[j]);
    }

    // Rotate by phi around X=intCoord[0]
    real3_rotation_matrix(R,intCoord[0],cosPhi,sinPhi);
    for (j=0; j<3; j++) {
      xNew[j]=real3_rotate<real3_x,real_x>(R,xNew[j]);
    }

    // Find theta, use lambda_AB=lambda_BA
    real3_x projVector=real3_cross(intCoord[2],real3_sub(xPrev[0],xPrev[2]));
    real_x denom=invMass[0]*real3_dot<real_x>(real3_sub(xPrev[0],xPrev[1]),projVector);
    real_x X_a2=real3_dot<real_x>(intCoord[0],xNew[0]);
    real_x Y_a2=real3_dot<real_x>(intCoord[1],xNew[0]);
    real_x gamma=real3_dot<real_x>(x[0],projVector)/denom; // constant coeffient
    real_x alpha=(X_a2*real3_dot<real_x>(intCoord[1],projVector)-Y_a2*real3_dot<real_x>(intCoord[0],projVector))/denom; // sin coefficient
    real_x beta=(X_a2*real3_dot<real_x>(intCoord[0],projVector)+Y_a2*real3_dot<real_x>(intCoord[1],projVector))/denom; // cos coefficient

    projVector=real3_cross(intCoord[2],real3_sub(xPrev[1],xPrev[2]));
    denom=invMass[1]*real3_dot<real_x>(real3_sub(xPrev[1],xPrev[0]),projVector);
    real_x X_b2=real3_dot<real_x>(intCoord[0],xNew[1]);
    real_x Y_b2=real3_dot<real_x>(intCoord[1],xNew[1]);
    gamma-=real3_dot<real_x>(x[1],projVector)/denom; // constant coeffient
    alpha-=(X_b2*real3_dot<real_x>(intCoord[1],projVector)-Y_b2*real3_dot<real_x>(intCoord[0],projVector))/denom; // sin coefficient
    beta-=(X_b2*real3_dot<real_x>(intCoord[0],projVector)+Y_b2*real3_dot<real_x>(intCoord[1],projVector))/denom; // cos coefficient

    // alpha*sin(theta)+beta*cos(theta)=gamma
    real_x signA=(alpha>0?1:-1);
    real_x sinTheta=(alpha*gamma-signA*beta*sqrt(alpha*alpha+beta*beta-gamma*gamma))/(alpha*alpha+beta*beta);
    real_x cosTheta=sqrt(1-sinTheta*sinTheta);

    // Rotate by theta around Z=intCoord[2]
    real3_rotation_matrix(R,intCoord[2],cosTheta,sinTheta);
    for (j=0; j<3; j++) {
      xNew[j]=real3_rotate<real3_x,real_x>(R,xNew[j]);
    }

    // Finish up
    for (j=0; j<3; j++) {
      real3_inc(&xPrev[j],comPrev);
      real3_inc(&xNew[j],com);
      ((real3_v*)ls.v)[cons[i].idx[j]]=real3_scale<real3_v>(1/(real_x)lp.halfdt,real3_sub(xNew[j],xPrev[j]));
      ((real3_x*)ls.x)[cons[i].idx[j]]=real3_sub(xNew[j],xShift[j]);
    }
  }
}

// __global__ void holonomic_position_branch1_kernel(int N,struct Branch1Cons *cons,struct LeapState ls,struct LeapParms2 lp,real_x *xPrevious,real3_x box)
// {}
//   int i=blockIdx.x*blockDim.x+threadIdx.x;
template <bool flagBox,typename box_type>
__device__ inline void holonomic_position_branch1_kernel(int N,int B0,struct Branch1Cons *cons,struct LeapState ls,struct LeapParms2 lp,real_x *xPrevious,box_type box,real tolerance)
{
  int i=(blockIdx.x-B0)*blockDim.x+threadIdx.x;
  real_x invMass[2];
  real3_x xPrev[2];
  real3_x x[2];
  real3_x xShift[2];
  real b02;
  int j;

  if (i<N) {
    b02=cons[i].b0[0];
    b02*=b02;

    for (j=0; j<2; j++) {
      xPrev[j]=((real3_x*)xPrevious)[cons[i].idx[j]];
      if (j>0) {
        xShift[j]=real3_sub(real3_subpbc<flagBox>(xPrev[j],xPrev[0],box),real3_sub(xPrev[j],xPrev[0]));
      } else {
        xShift[0]=real3_reset<real3_x>();
      }
      xPrev[j]=real3_add(xPrev[j],xShift[j]);
      x[j]=real3_add(((real3_x*)ls.x)[cons[i].idx[j]],xShift[j]);
      invMass[j]=ls.ism[3*cons[i].idx[j]];
      invMass[j]*=invMass[j];
    }

    real3_x r12Prev=real3_sub(xPrev[0],xPrev[1]);
    real3_x r12=real3_sub(x[0],x[1]);
    real_x r12PrevDotR12=real3_dot<real_x>(r12Prev,r12);
    real_x lambda=(-r12PrevDotR12+sqrt(r12PrevDotR12*r12PrevDotR12-real3_mag2<real_x>(r12Prev)*(real3_mag2<real_x>(r12)-b02)))/(real3_mag2<real_x>(r12Prev)*(invMass[0]+invMass[1]));

    real3_scaleinc(&x[0], invMass[0]*lambda,r12Prev);
    real3_scaleinc(&x[1],-invMass[1]*lambda,r12Prev);
  
    // Finish up
    for (j=0; j<2; j++) {
      ((real3_v*)ls.v)[cons[i].idx[j]]=real3_scale<real3_v>(1/(real_x)lp.halfdt,real3_sub(x[j],xPrev[j]));
      ((real3_x*)ls.x)[cons[i].idx[j]]=real3_sub(x[j],xShift[j]);
    }
  }
}

// __global__ void holonomic_position_branch2_kernel(int N,struct Branch2Cons *cons,struct LeapState ls,struct LeapParms2 lp,real_x *xPrevious,real3_x box,real tolerance)
// {}
//   int i=blockIdx.x*blockDim.x+threadIdx.x;
template <bool flagBox,typename box_type>
__device__ inline void holonomic_position_branch2_kernel(int N,int B0,struct Branch2Cons *cons,struct LeapState ls,struct LeapParms2 lp,real_x *xPrevious,box_type box,real tolerance)
{
  int i=(blockIdx.x-B0)*blockDim.x+threadIdx.x;
  real_x invMass[3];
  real3_x xPrev[3];
  real3_x x[3];
  real3_x xShift[3];
  real3_x dxPrev[2];
  real3_x dx[2];
  real b02[2];
  real_x AT[2][2];
  real_x b[2];
  real_x lambda[2];
  int j,jp;
  bool unfinished;
  int iteration;

  if (i<N) {
    for (j=0; j<3; j++) {
      xPrev[j]=((real3_x*)xPrevious)[cons[i].idx[j]];
      if (j>0) {
        xShift[j]=real3_sub(real3_subpbc<flagBox>(xPrev[j],xPrev[0],box),real3_sub(xPrev[j],xPrev[0]));
      } else {
        xShift[0]=real3_reset<real3_x>();
      }
      xPrev[j]=real3_add(xPrev[j],xShift[j]);
      x[j]=real3_add(((real3_x*)ls.x)[cons[i].idx[j]],xShift[j]);
      invMass[j]=ls.ism[3*cons[i].idx[j]];
      invMass[j]*=invMass[j];
    }

    for (j=0; j<2; j++) {
      b02[j]=cons[i].b0[j];
      b02[j]*=b02[j];
      dxPrev[j]=real3_sub(xPrev[0],xPrev[j+1]);
    }

    iteration=-1;

    iteration++;
    unfinished=false;
    for (j=0; j<2; j++) {
      dx[j]=real3_sub(x[0],x[j+1]);
      b[j]=b02[j]-real3_mag2<real_x>(dx[j]);
      if (abs(b[j])>=2*b02[j]*tolerance) unfinished=true;
    }

    while (unfinished && iteration<MAXITERATION) {
      for (j=0; j<2; j++) {
        AT[0][j]=2*real3_dot<real_x>(dxPrev[0],dx[j])*invMass[0];
        AT[1][j]=2*real3_dot<real_x>(dxPrev[1],dx[j])*invMass[0];
        AT[j][j]+=2*real3_dot<real_x>(dxPrev[j],dx[j])*invMass[j+1];
      }

      real_x invDet=1/(AT[0][0]*AT[1][1]-AT[0][1]*AT[1][0]);
      for (j=0; j<2; j++) {
        jp=1-j;
        lambda[j]=(b[j]*AT[jp][jp]-b[jp]*AT[jp][j])*invDet;
      }

      for (j=0; j<2; j++) {
        real3_scaleinc(&x[0],lambda[j]*invMass[0],dxPrev[j]);
        real3_scaleinc(&x[j+1],-lambda[j]*invMass[j+1],dxPrev[j]);
      }

      iteration++;
      unfinished=false;
      for (j=0; j<2; j++) {
        dx[j]=real3_sub(x[0],x[j+1]);
        b[j]=b02[j]-real3_mag2<real_x>(dx[j]);
        if (abs(b[j])>=2*b02[j]*tolerance) unfinished=true;
      }
    }

    // Finish up
    for (j=0; j<3; j++) {
      ((real3_v*)ls.v)[cons[i].idx[j]]=real3_scale<real3_v>(1/(real_x)lp.halfdt,real3_sub(x[j],xPrev[j]));
      ((real3_x*)ls.x)[cons[i].idx[j]]=real3_sub(x[j],xShift[j]);
    }
  }
}

// __global__ void holonomic_position_branch3_kernel(int N,struct Branch3Cons *cons,struct LeapState ls,struct LeapParms2 lp,real_x *xPrevious,real3_x box,real tolerance)
// {}
//   int i=blockIdx.x*blockDim.x+threadIdx.x;
template <bool flagBox,typename box_type>
__device__ inline void holonomic_position_branch3_kernel(int N,int B0,struct Branch3Cons *cons,struct LeapState ls,struct LeapParms2 lp,real_x *xPrevious,box_type box,real tolerance)
{
  int i=(blockIdx.x-B0)*blockDim.x+threadIdx.x;
  real_x invMass[4];
  real3_x xPrev[4];
  real3_x x[4];
  real3_x xShift[4];
  real3_x dxPrev[3];
  real3_x dx[3];
  real b02[3];
  real_x AT[3][3];
  real_x b[3];
  real_x lambda[3];
  int j,jp,jpp;
  bool unfinished;
  int iteration;

  if (i<N) {
    for (j=0; j<4; j++) {
      xPrev[j]=((real3_x*)xPrevious)[cons[i].idx[j]];
      if (j>0) {
        xShift[j]=real3_sub(real3_subpbc<flagBox>(xPrev[j],xPrev[0],box),real3_sub(xPrev[j],xPrev[0]));
      } else {
        xShift[0]=real3_reset<real3_x>();
      }
      xPrev[j]=real3_add(xPrev[j],xShift[j]);
      x[j]=real3_add(((real3_x*)ls.x)[cons[i].idx[j]],xShift[j]);
      invMass[j]=ls.ism[3*cons[i].idx[j]];
      invMass[j]*=invMass[j];
    }

    for (j=0; j<3; j++) {
      b02[j]=cons[i].b0[j];
      b02[j]*=b02[j];
      dxPrev[j]=real3_sub(xPrev[0],xPrev[j+1]);
    }

    iteration=-1;

    iteration++;
    unfinished=false;
    for (j=0; j<3; j++) {
      dx[j]=real3_sub(x[0],x[j+1]);
      b[j]=b02[j]-real3_mag2<real_x>(dx[j]);
      if (abs(b[j])>=2*b02[j]*tolerance) unfinished=true;
    }

    while (unfinished && iteration<MAXITERATION) {
      for (j=0; j<3; j++) {
        AT[0][j]=2*real3_dot<real_x>(dxPrev[0],dx[j])*invMass[0];
        AT[1][j]=2*real3_dot<real_x>(dxPrev[1],dx[j])*invMass[0];
        AT[2][j]=2*real3_dot<real_x>(dxPrev[2],dx[j])*invMass[0];
        AT[j][j]+=2*real3_dot<real_x>(dxPrev[j],dx[j])*invMass[j+1];
      }

      for (j=0; j<3; j++) {
        // jp=(j+1)%3;
        // jpp=(j+2)%3;
        jp=((j==2)?0:(j+1));
        jpp=3-(j+jp);
        real3_x partialDet=real3_cross(((real3_x*)AT)[jp],((real3_x*)AT)[jpp]);
        lambda[j]=real3_dot<real_x>(((real3_x*)b)[0],partialDet)/real3_dot<real_x>(((real3_x*)AT)[j],partialDet);
      }

      for (j=0; j<3; j++) {
        real3_scaleinc(&x[0],lambda[j]*invMass[0],dxPrev[j]);
        real3_scaleinc(&x[j+1],-lambda[j]*invMass[j+1],dxPrev[j]);
      }

      iteration++;
      unfinished=false;
      for (j=0; j<3; j++) {
        dx[j]=real3_sub(x[0],x[j+1]);
        b[j]=b02[j]-real3_mag2<real_x>(dx[j]);
        if (abs(b[j])>=2*b02[j]*tolerance) unfinished=true;
      }
    }

    // Finish up
    for (j=0; j<4; j++) {
      ((real3_v*)ls.v)[cons[i].idx[j]]=real3_scale<real3_v>(1/(real_x)lp.halfdt,real3_sub(x[j],xPrev[j]));
      ((real3_x*)ls.x)[cons[i].idx[j]]=real3_sub(x[j],xShift[j]);
    }
  }
}

/*
__global__ void holonomic_position_alttriangle_kernel(int N,struct TriangleCons *cons,struct LeapState ls,struct LeapParms2 lp,real_x *xPrevious,real3_x box,real tolerance)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
// __device__ inline void holonomic_position_alttriangle_kernel(int N,int B0,struct TriangleCons *cons,struct LeapState ls,struct LeapParms2 lp,real_x *xPrevious,real3_x box,real tolerance)
// {}
//   int i=(blockIdx.x-B0)*blockDim.x+threadIdx.x;
  real_x invMass[3];
  real3_x xPrev[3];
  real3_x x[3];
  real3_x xShift[3];
  real3_x dxPrev[3];
  real3_x dx[3];
  real b02[3];
  real_x AT[3][3];
  real_x b[3];
  real_x lambda[3];
  int j,jp,jpp;
  bool unfinished;
  int iteration;

  if (i<N) {
    for (j=0; j<3; j++) {
      xPrev[j]=((real3_x*)xPrevious)[cons[i].idx[j]];
      if (j>0) {
        xShift[j]=real3_sub(real3_subpbc(xPrev[j],xPrev[0],box),real3_sub(xPrev[j],xPrev[0]));
      } else {
        xShift[0]=real3_reset<real3_x>();
      }
      xPrev[j]=real3_add(xPrev[j],xShift[j]);
      x[j]=real3_add(((real3_x*)ls.x)[cons[i].idx[j]],xShift[j]);
      invMass[j]=ls.ism[3*cons[i].idx[j]];
      invMass[j]*=invMass[j];
    }

    // b0 is in 01 02 12 order. j is in 01 12 20 order
    for (j=0; j<3; j++) {
      jpp=((j==0)?0:(3-j));
      b02[j]=cons[i].b0[jpp];
      b02[j]*=b02[j];
      jp=((j==2)?0:(j+1));
      dxPrev[j]=real3_sub(xPrev[j],xPrev[jp]);
    }

    iteration=-1;

    iteration++;
    unfinished=false;
    for (j=0; j<3; j++) {
      jp=((j==2)?0:(j+1));
      dx[j]=real3_sub(x[j],x[jp]);
      b[j]=b02[j]-real3_mag2<real_x>(dx[j]);
      if (abs(b[j])>=2*b02[j]*tolerance) unfinished=true;
    }

    while (unfinished && iteration<MAXITERATION) {
      for (j=0; j<3; j++) {
        jp=((j==2)?0:(j+1));
        jpp=3-(j+jp);
        // AT is A transpose...
        AT[j][j]=2*real3_dot<real_x>(dxPrev[j],dx[j])*invMass[j];
        AT[j][j]+=2*real3_dot<real_x>(dxPrev[j],dx[j])*invMass[jp];
        AT[jp][j]=-2*real3_dot<real_x>(dxPrev[jp],dx[j])*invMass[jp];
        AT[jpp][j]=-2*real3_dot<real_x>(dxPrev[jpp],dx[j])*invMass[j];
      }

      for (j=0; j<3; j++) {
        // jp=(j+1)%3;
        // jpp=(j+2)%3;
        jp=((j==2)?0:(j+1));
        jpp=3-(j+jp);
        real3_x partialDet=real3_cross(((real3_x*)AT)[jp],((real3_x*)AT)[jpp]);
        lambda[j]=real3_dot<real_x>(((real3_x*)b)[0],partialDet)/real3_dot<real_x>(((real3_x*)AT)[j],partialDet);
      }

      for (j=0; j<3; j++) {
        jp=((j==2)?0:(j+1));
        real3_scaleinc(&x[j],lambda[j]*invMass[j],dxPrev[j]);
        real3_scaleinc(&x[jp],-lambda[j]*invMass[jp],dxPrev[j]);
      }

      iteration++;
      unfinished=false;
      for (j=0; j<3; j++) {
        jp=((j==2)?0:(j+1));
        dx[j]=real3_sub(x[j],x[jp]);
        b[j]=b02[j]-real3_mag2<real_x>(dx[j]);
        if (abs(b[j])>=2*b02[j]*tolerance) unfinished=true;
      }
    }

    // Finish up
    for (j=0; j<3; j++) {
      ((real3_v*)ls.v)[cons[i].idx[j]]=real3_scale<real3_v>(1/(real_x)lp.halfdt,real3_sub(x[j],xPrev[j]));
      ((real3_x*)ls.x)[cons[i].idx[j]]=real3_sub(x[j],xShift[j]);
    }
  }
}
*/

template <bool flagBox,typename box_type>
__global__ void holonomic_position_kernel(int N0,int N1,int N2,int N3,int B0,int B1,int B2,int B3,struct TriangleCons *cons0,struct Branch1Cons *cons1,struct Branch2Cons *cons2,struct Branch3Cons *cons3,struct LeapState ls,struct LeapParms2 lp,real_x *xPrevious,box_type box,real tolerance)
{
  if (blockIdx.x<B0) {
    holonomic_position_triangle_kernel<flagBox>(N0,0,cons0,ls,lp,xPrevious,box,tolerance);
  } else if (blockIdx.x<B1) {
    holonomic_position_branch1_kernel<flagBox>(N1,B0,cons1,ls,lp,xPrevious,box,tolerance);
  } else if (blockIdx.x<B2) {
    holonomic_position_branch2_kernel<flagBox>(N2,B1,cons2,ls,lp,xPrevious,box,tolerance);
  } else if (blockIdx.x<B3) {
    holonomic_position_branch3_kernel<flagBox>(N3,B2,cons3,ls,lp,xPrevious,box,tolerance);
  }
}

template <bool flagBox,typename box_type>
void holonomic_positionT(System *system,box_type box)
{
  Run *r=system->run;
  State *s=system->state;
  Potential *p=system->potential;
/*  int N;

  N=p->triangleConsCount;
  if (N) holonomic_position_triangle_kernel<<<(N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(N,p->triangleCons_d,s->leapState[0],s->leapParms2[0],s->positionCons_d,s->orthBox);

  N=p->branch1ConsCount;
  if (N) holonomic_position_branch1_kernel<<<(N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(N,p->branch1Cons_d,s->leapState[0],s->leapParms2[0],s->positionCons_d,s->orthBox);

  N=p->branch2ConsCount;
  if (N) holonomic_position_branch2_kernel<<<(N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(N,p->branch2Cons_d,s->leapState[0],s->leapParms2[0],s->positionCons_d,s->orthBox,r->shakeTolerance);

  N=p->branch3ConsCount;
  if (N) holonomic_position_branch3_kernel<<<(N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(N,p->branch3Cons_d,s->leapState[0],s->leapParms2[0],s->positionCons_d,s->orthBox,r->shakeTolerance);*/

  int N0,N1,N2,N3;
  int B0,B1,B2,B3;
  N0=p->triangleConsCount;
  N1=p->branch1ConsCount;
  N2=p->branch2ConsCount;
  N3=p->branch3ConsCount;
  B0=(N0+BLUP-1)/BLUP;
  B1=B0+(N1+BLUP-1)/BLUP;
  B2=B1+(N2+BLUP-1)/BLUP;
  B3=B2+(N3+BLUP-1)/BLUP;

  if (N0+N1+N2+N3) holonomic_position_kernel<flagBox><<<B3,BLUP,0,r->updateStream>>>(N0,N1,N2,N3,B0,B1,B2,B3,p->triangleCons_d,p->branch1Cons_d,p->branch2Cons_d,p->branch3Cons_d,s->leapState[0],s->leapParms2[0],s->positionCons_d,box,r->shakeTolerance);
}

void holonomic_position(System *system)
{
  if (system->state->typeBox) {
    holonomic_positionT<true>(system,system->state->tricBox);
  } else {
    holonomic_positionT<false>(system,system->state->orthBox);
  }
}

void holonomic_backup_position(LeapState *leapState,real_x *positionCons,cudaStream_t stream)
{
  if (positionCons) {
    cudaMemcpyAsync(positionCons,leapState->x,leapState->N*sizeof(real_x),cudaMemcpyDeviceToDevice,stream);
  }
}
