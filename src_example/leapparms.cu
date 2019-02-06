
#include "leapparms.h"

#include "defines.h"

#include <math.h>
#include <stdlib.h>

static
void calcABCD(real slambda,real* A,real* B,real* C,real* D)
{
  real sign,shlambda;
  sign = (slambda>0 ? 1.0 : -1.0);

  if ((slambda/sign)>0.12) {
    // V noise variance
    A[0]=sign*(exp(slambda)-1);
    // related to conditional variance
    B[0]=slambda*(exp(slambda)-1)-4*(exp(slambda/2)-1)*(exp(slambda/2)-1);
    // X noise variance
    C[0]=sign*(slambda-3+4*exp(-slambda/2)-exp(-slambda));
    // VX noise correlation
    D[0]=sign*(2-exp(slambda/2)-exp(-slambda/2));
  } else {
    shlambda=slambda/2;
    // Max As error: 2e-14 at shlambda=0.025; error(0.1)=3e-10
    // As=sign*(2*shlambda+2*shlambda^2+(4./3)*shlambda^3+(2./3)*shlambda^4+(4./15)*shlambda^5+(4./45)*shlambda^6+(8./315)*shlambda^7);
    A[0]=(8./315); // shl^7
    A[0]=(4./45)+A[0]*shlambda; // shl^6
    A[0]=(4./15)+A[0]*shlambda; // shl^5
    A[0]=(2./3)+A[0]*shlambda; // shl^4
    A[0]=(4./3)+A[0]*shlambda; // shl^3
    A[0]=2+A[0]*shlambda; // shl^2
    A[0]=2+A[0]*shlambda; // shl
    A[0]=sign*A[0]*shlambda;
    // Max Bs error: 2e-11 at shlambda=0.08; error(0.1)=1e-10
    // Bs=(1./3)*shlambda^4+(1./3)*shlambda^5+(17./90)*shlambda^6+(7./90)*shlambda^7+(43./1680)*shlambda^8+(107./15120)*shlambda^9+(769./453600)*shlambda^10;
    B[0]=(769./453600); // shl^10;
    B[0]=(107./15120)+B[0]*shlambda; // shl^9
    B[0]=(43./1680)+B[0]*shlambda; // shl^8
    B[0]=(7./90)+B[0]*shlambda; // shl^7
    B[0]=(17./90)+B[0]*shlambda; // shl^6
    B[0]=(1./3)+B[0]*shlambda; // shl^5
    B[0]=(1./3)+B[0]*shlambda; // shl^4
    B[0]=B[0]*shlambda*shlambda*shlambda*shlambda;
    // Max Ds error: 2e-14 at shlambda=0.12; error(0.1)=5e-14
    // Ds=sign*(-shlambda^2-(1./12)*shlambda^4-(1./360)*shlambda^6-(1./20160)*shlambda^8);
    D[0]=-(1./20160); // shl^8
    D[0]=-(1./360)+D[0]*shlambda*shlambda; // shl^6
    D[0]=-(1./12)+D[0]*shlambda*shlambda; // shl^4
    D[0]=-1+D[0]*shlambda*shlambda; // shl^2
    D[0]=sign*D[0]*shlambda*shlambda;
    // Best combined accuracy 1e-11 at shlambda=0.06, from semilogy(shlambda,max(max(abs(Ds1-Ds2)./Ds2,abs(Bs1-Bs2)./Bs2),abs(As1-As2)./As2))*/
  }
}


struct_leapparms* alloc_leapparms(real dt,real gamma,real kT)
{
  struct_leapparms *leapparms;
  real lambda;
  real As, Bs, Cs, Ds;
  real Ar, Br, Cr, Dr;

  leapparms=(struct_leapparms*) malloc(sizeof(struct_leapparms));

  // integrator from W. F. Van Gunsteren and H. J. C. Berendsen (1988)
  leapparms->dt=dt;
  leapparms->gamma=gamma;
  leapparms->kT=kT;
#warning "kTeff is set automatically"
  leapparms->kTeff=kT*(1.0+0.0032e6*dt*dt);
  leapparms->mDiff=kT/gamma; // Diffusion * Mass
  lambda=dt*gamma;
  // [t to t+dt/2]
  // Ar prop svv^2; Br=Ar*Cr-Dr^2; Cr prop sxx^2; Dr prop sxv^2
  calcABCD(lambda,&As,&Bs,&Cs,&Ds);
  // Eq 3.19
  leapparms->SigmaVsVs=sqrt(leapparms->mDiff*gamma*As);
  leapparms->XsVsCoeff=Ds/(gamma*As); // Equal to SigmaXsVs^2/SigmaVsVs^2
  leapparms->CondSigmaXsXs=sqrt((leapparms->mDiff/gamma)*Bs/As); // Equal to sqrt(SigmaXsXs^2-SigmaXsVs^4/SigmaVsVs^2)

  // [t+dt/2 to t+dt]
  calcABCD(-lambda,&Ar,&Br,&Cr,&Dr);
  leapparms->SigmaVrVr=sqrt(leapparms->mDiff*gamma*Ar);
  leapparms->XrVrCoeff=Dr/(gamma*Ar); // Equal to SigmaXrVr^2/SigmaVrVr^2
  leapparms->CondSigmaXrXr=sqrt((leapparms->mDiff/gamma)*Br/Ar); // Equal to sqrt(SigmaXrXr^2-SigmaXrVr^4/SigmaVrVr^2)

  // Made up
  leapparms->vhalf_vscale=exp(-lambda/2);
  leapparms->vhalf_ascale=dt*(1-exp(-lambda/2))/lambda;
  leapparms->vhalf_Vsscale=-exp(-lambda/2);

  // Eq 3.6
  leapparms->v_vscale=exp(-lambda);
  leapparms->v_ascale=dt*Ar/lambda;
  leapparms->v_Vsscale=-exp(-lambda);

  // Eq 3.10
  leapparms->x_vscale=dt*exp(lambda/2)*Ar/lambda;
  leapparms->x_Xrscale=-1;

  return leapparms;
}


void free_leapparms(struct_leapparms* leapparms)
{
  free(leapparms);
}

