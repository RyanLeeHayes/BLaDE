#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

#include "system/potential.h"
#include "system/system.h"
#include "io/io.h"
#include "system/parameters.h"
#include "system/structure.h"
#include "msld/msld.h"
#include "system/state.h"
#include "run/run.h"
#include "domdec/domdec.h"
#include "rng/rng_gpu.h"
#include "bonded/bonded.h"
#include "bonded/pair.h"
#include "nbrecip/nbrecip.h"
#include "nbdirect/nbdirect.h"
#include "restrain/restrain.h"
#include "holonomic/virtual.h"

#ifdef USE_TEXTURE
#include <string.h> // for memset
#endif



// Utility functions
bool operator<(const CountType& a,const CountType& b)
{
  // Sort in descending order type count, not ascending
  return a.count>b.count || (a.count==b.count && a.type<b.type);
}

// Class constructors
Potential::Potential() {
  atomCount=0;
  bondCount=0;
  bond12Count=0;
  bond13Count=0;
  bonds=NULL;
  bonds_d=NULL;
  angleCount=0;
  angles=NULL;
  angles_d=NULL;
  diheCount=0;
  dihes=NULL;
  dihes_d=NULL;
  imprCount=0;
  imprs=NULL;
  imprs_d=NULL;
  cmapCount=0;
  cmaps=NULL;
  cmaps_d=NULL;

  softBondCount=0;
  softBonds=NULL;
  softBonds_d=NULL;
  softAngleCount=0;
  softAngles=NULL;
  softAngles_d=NULL;
  softDiheCount=0;
  softDihes=NULL;
  softDihes_d=NULL;
  softImprCount=0;
  softImprs=NULL;
  softImprs_d=NULL;
  softCmapCount=0;
  softCmaps=NULL;
  softCmaps_d=NULL;

  charge=NULL;
  charge_d=NULL;

  nb14Count=0;
  nb14s=NULL;
  nb14s_d=NULL;
  nbexCount=0;
  nbexs=NULL;
  nbexs_d=NULL;

  virtExcl=NULL;
  bondExcl=NULL;
  angleExcl=NULL;
  diheExcl=NULL;
  msldExcl=NULL;
  allExcl=NULL;

  exclCount=0;
  excls=NULL;
  excls_d=NULL;

  chargeGridPME_d=NULL;
  fourierGridPME_d=NULL;
  potentialGridPME_d=NULL;
#ifdef USE_TEXTURE
  potentialGridPME_tex=0;
#endif
  planFFTPME=0;
  planIFFTPME=0;

  nbonds=NULL;
  nbonds_d=NULL;
  vdwParameterCount=0;
  vdwParameters=NULL;
  vdwParameters_d=NULL;
#ifdef USE_TEXTURE
  vdwParameters_tex=0;
#endif

  triangleConsCount=0;
  triangleCons=NULL;
  triangleCons_d=NULL;
  branch1ConsCount=0;
  branch1Cons=NULL;
  branch1Cons_d=NULL;
  branch2ConsCount=0;
  branch2Cons=NULL;
  branch2Cons_d=NULL;
  branch3ConsCount=0;
  branch3Cons=NULL;
  branch3Cons_d=NULL;

  noeCount=0;
  noes=NULL;
  noes_d=NULL;
  harmCount=0;
  harms=NULL;
  harms_d=NULL;

  prettifyPlan=NULL;
}

Potential::~Potential()
{
  if (bonds) free(bonds);
  if (angles) free(angles);
  if (dihes) free(dihes);
  if (imprs) free(imprs);
  if (cmaps) free(cmaps);
  if (bonds_d) cudaFree(bonds_d);
  if (angles_d) cudaFree(angles_d);
  if (dihes_d) cudaFree(dihes_d);
  if (imprs_d) cudaFree(imprs_d);
  if (cmaps_d) cudaFree(cmaps_d);

  if (softBonds) free(softBonds);
  if (softAngles) free(softAngles);
  if (softDihes) free(softDihes);
  if (softImprs) free(softImprs);
  if (softCmaps) free(softCmaps);
  if (softBonds_d) cudaFree(softBonds_d);
  if (softAngles_d) cudaFree(softAngles_d);
  if (softDihes_d) cudaFree(softDihes_d);
  if (softImprs_d) cudaFree(softImprs_d);
  if (softCmaps_d) cudaFree(softCmaps_d);

  for (std::map<TypeName8O,real(*)[4][4]>::iterator ii=cmapTypeToPtr.begin(); ii!=cmapTypeToPtr.end(); ii++) {
    cudaFree(ii->second);
  }
  cmapTypeToPtr.clear();
  for (std::map<TypeName8O,real(*)[4][4]>::iterator ii=cmapRestTypeToPtr.begin(); ii!=cmapRestTypeToPtr.end(); ii++) {
    cudaFree(ii->second);
  }
  cmapRestTypeToPtr.clear();

  if (charge) free(charge);
  if (charge_d) cudaFree(charge_d);

  if (nb14s) free(nb14s);
  if (nb14s_d) cudaFree(nb14s_d);
  if (nbexs) free(nbexs);
  if (nbexs_d) cudaFree(nbexs_d);

  delete [] virtExcl;
  delete [] bondExcl;
  delete [] angleExcl;
  delete [] diheExcl;
  delete [] msldExcl;
  delete [] allExcl;

  if (excls) free(excls);
  if (excls_d) cudaFree(excls_d);

  if (chargeGridPME_d) cudaFree(chargeGridPME_d);
  if (fourierGridPME_d) cudaFree(fourierGridPME_d);
  if (potentialGridPME_d) cudaFree(potentialGridPME_d);
#ifdef USE_TEXTURE
  if (potentialGridPME_tex) cudaDestroyTextureObject(potentialGridPME_tex);
#endif

  if (nbonds) free(nbonds);
  if (nbonds_d) cudaFree(nbonds_d);
  if (vdwParameters) free(vdwParameters);
  if (vdwParameters_d) cudaFree(vdwParameters_d);
#ifdef USE_TEXTURE
  if (vdwParameters_tex) cudaDestroyTextureObject(vdwParameters_tex);
#endif

  if (triangleCons) free(triangleCons);
  if (triangleCons_d) cudaFree(triangleCons_d);
  if (branch1Cons) free(branch1Cons);
  if (branch1Cons_d) cudaFree(branch1Cons_d);
  if (branch2Cons) free(branch2Cons);
  if (branch2Cons_d) cudaFree(branch2Cons_d);
  if (branch3Cons) free(branch3Cons);
  if (branch3Cons_d) cudaFree(branch3Cons_d);

  if (planFFTPME) cufftDestroy(planFFTPME);
  if (planIFFTPME) cufftDestroy(planIFFTPME);

  if (noes) free(noes);
  if (noes_d) cudaFree(noes_d);
  if (harms) free(harms);
  if (harms_d) cudaFree(harms_d);

  if (prettifyPlan) free(prettifyPlan);
}



// Methods
// Bicubic spline interpolation is used for CMAP interactions. Bicubic spline interpolation is a special case of bicubic interpolation, where cubic splines are used to obtain the derivatives requires for bicubic interpolation. The function below, cmap_cubic_spline_interpolate can be used to interpolate for function values and slopes once splines have been set up with cmap_cubic_spline_setup, but since bicubic interpolation only needs slopes at the vertices, setting up the splines with cmap_cubic_spline_setup is sufficient, and this function is never called.
// Input:
// ngrid - number of spline/grid points
// xin - input value for interpolation
// dx - uniform spacing of grid points
// yin - vector of length ngrid of the target values as the grid points
// dyin - vector of length ngrid of the slopes at grid points (computed elsewhere)
// Output:
// yout - the interpolated value of the function
// dyout - inter interpolated slope of the function
void cmap_cubic_spline_interpolate(int ngrid,real xin,real dx,real *yin,real *dyin,real *yout,real *dyout)
{
  int xint0,xint1;
  real remainder;
  int u,v;

  xin/=dx; // Function only operates with unit scaling, you need to multiply by 1/dx outside if you want the real derivative.
  xin=fmod(xin,ngrid);
  xin+=(xin<0)?ngrid:0;
  xint0=((int)floor(xin))%ngrid;
  xint1=(xint0+1)%ngrid;
  remainder=xin-xint0;

  u=remainder;
  v=1-remainder;

  yout[0]=yin[xint0]*v*v*(3*u+v) + dyin[xint0]*u*v*v
    - dyin[xint1]*u*u*v + yin[xint1]*u*u*(3*v+u);
  dyout[0]=yin[xint0]*-6*u*v + dyin[xint0]*v*(v-2*u)
    + dyin[xint1]*u*(u-2*v) + yin[xint1]*6*u*v;
}

// cubic splines are piecewise cubic polynomicals with the constraint that functions go through target point, and interpolated functions have continuous first and second derivatives. A cubic spline is most easily represented in terms of the values of the function at points (which are given), and the slope of the function at the grid points (which is determined by this function. In the CHARMM implementation of CMAP, rather than correctly treating the periodic boundary conditions, a buffer of ngrid/2 extra points is added on each side to mitigate artifacts due to edge effects. Errors due to that approximation appear to be one part in 10^6 to 10^7 for standard grids of 24 points. Periodic boundary conditions are correctly treated here.
// Input:
// ngrid - number of uniformly spaced points
// y - target function values
// Output:
// dy - the slopes at the grid points necessary to satisfy the cubic spline definition.
// Algorithm:
// The problem to be solved can be expressed in matrix notation as
// [ 4  1  0  ... 0  0  1 ]   [  dy[0]  ]   [ 3*(y[1]-y[N-1]) ]
// [ 1  4  1  ... 0  0  0 ]   [  dy[1]  ]   [  3*(y[2]-y[0])  ]
// [ 0  1  4  ... 0  0  0 ]   [  dy[2]  ]   [  3*(y[3]-y[1])  ]
// [   ...    ...   ...   ] X [   ...   ] = [       ...       ]
// [ 0  0  0  ... 4  1  0 ]   [ dy[N-3] ]   [3*(y[N-2]-y[N-4])]
// [ 0  0  0  ... 1  4  1 ]   [ dy[N-2] ]   [3*(y[N-1]-y[N-3])]
// [ 1  0  0  ... 0  1  4 ]   [ dy[N-1] ]   [ 3*(y[0]-y[N-2]) ]
// Using non-periodic boundary conditions changes the first and last rows
void cmap_cubic_spline_setup(int ngrid,real *y,real *dy)
{
  real diag[ngrid]; // diagonal corners of reduced matrix
  real offDiag[ngrid]; // off diagonal corners of reduced matrix
  int i,im1,ip1;
  real det,dy0,dy1;

  diag[ngrid-1]=4;
  offDiag[ngrid-1]=1;

  for (i=0; i<ngrid; i++) {
    im1=(i+ngrid-1)%ngrid;
    ip1=(i+ngrid+1)%ngrid;
    dy[i]=3*(y[ip1]-y[im1]);
  }

  for (i=(ngrid-2); i>0; i--) {
    offDiag[i]=-offDiag[i+1]/diag[i+1];
    diag[i]=diag[i+1]+offDiag[i+1]*offDiag[i];
    dy[0]+=dy[i+1]*offDiag[i];
    dy[i]+=dy[i+1]*(-1/diag[i+1]);
  }

  det=diag[1]*diag[1]-(1+offDiag[1])*(1+offDiag[1]);
  dy0=(diag[1]*dy[0]-(1+offDiag[1])*dy[1])/det;
  dy1=(diag[1]*dy[1]-(1+offDiag[1])*dy[0])/det;
  dy[0]=dy0;
  dy[1]=dy1;

  for (i=2; i<ngrid; i++) {
    dy[i]=(dy[i]-dy[i-1]-offDiag[i]*dy[0])/diag[i];
  }
}

// Convert f (first element of intermediate), fx (second element of intermediate), fy (third element of intermediate), and fxy (fourth element of intermediate to a[4][4] which is stored in cmap parameters. Procedure is noted both in CHARMM source code, on wikipedia https://en.wikipedia.org/wiki/Bicubic_interpolation and in various other references. x=[f00,f10,f01,f11,fx00,fx10,fx01,fx11,fy00,fy10,fy01,fy11,fxy00,fxy10,fxy01,fxy11], and a=[a00,a10,a20,a30,a01,a11,a21,a31,a02,a12,a22,a32,a03,a13,a23,a33] -- gives me the jeebies, that's fortran ordering.
// a=Ainv*x
// F(phi,psi)=sum_{i=0 to 3} sum_{j=0 to 3} aij*phi^i*psi^j
void bicubic_setup(int ngrid,real (*cmapIntermediate)[4],real (*cmapParameters)[4][4])
{
  real Ainv[16][16]={
    { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {-3, 3, 0, 0,-2,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    { 2,-2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
    { 0, 0, 0, 0, 0, 0, 0, 0,-3, 3, 0, 0,-2,-1, 0, 0},
    { 0, 0, 0, 0, 0, 0, 0, 0, 2,-2, 0, 0, 1, 1, 0, 0},
    {-3, 0, 3, 0, 0, 0, 0, 0,-2, 0,-1, 0, 0, 0, 0, 0},
    { 0, 0, 0, 0,-3, 0, 3, 0, 0, 0, 0, 0,-2, 0,-1, 0},
    { 9,-9,-9, 9, 6, 3,-6,-3, 6,-6, 3,-3, 4, 2, 2, 1},
    {-6, 6, 6,-6,-3,-3, 3, 3,-4, 4,-2, 2,-2,-2,-1,-1},
    { 2, 0,-2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0},
    { 0, 0, 0, 0, 2, 0,-2, 0, 0, 0, 0, 0, 1, 0, 1, 0},
    {-6, 6, 6,-6,-4,-2, 4, 2,-3, 3,-3, 3,-2,-1,-2,-1},
    { 4,-4,-4, 4, 2, 2,-2,-2, 2,-2, 2,-2, 1, 1, 1, 1}};
  real x[16], a[16];
  int i,j,k,l;
  int ip1,jp1;

  for (i=0; i<ngrid; i++) {
    ip1=(i+1)%ngrid;
    for (j=0; j<ngrid; j++) {
      jp1=(j+1)%ngrid;
      for (k=0; k<4; k++) {
        x[4*k+0]=cmapIntermediate[ngrid*i  +j  ][k];
        x[4*k+1]=cmapIntermediate[ngrid*ip1+j  ][k];
        x[4*k+2]=cmapIntermediate[ngrid*i  +jp1][k];
        x[4*k+3]=cmapIntermediate[ngrid*ip1+jp1][k];
      }
      for (k=0; k<16; k++) {
        a[k]=0;
        for (l=0; l<16; l++) {
          a[k]+=Ainv[k][l]*x[l];
        }
      }
      for (k=0; k<4; k++) {
        for (l=0; l<4; l++) {
          cmapParameters[ngrid*i+j][k][l]=a[k+4*l];
        }
      }
    }
  }
}

// Another method to get the same parameters. (Used to check for correctness during development).
void bicubic_setup_alternative(int ngrid,real (*cmapIntermediate)[4],real (*cmapParameters)[4][4])
{
  real T[4][4]={
    { 1, 0, 0, 0},
    { 0, 0, 1, 0},
    {-3, 3,-2,-1},
    { 2,-2, 1, 1}};
  real P[4][4]; // intermediate product
  real f[4][4];
  int i,j,k,l,m;
  int ip1,jp1;

  for (i=0; i<ngrid; i++) {
    ip1=(i+1)%ngrid;
    for (j=0; j<ngrid; j++) {
      jp1=(j+1)%ngrid;
      for (k=0; k<2; k++) {
        for (l=0; l<2; l++) {
          f[2*k+0][2*l+0]=cmapIntermediate[ngrid*i  +j  ][k+2*l];
          f[2*k+0][2*l+1]=cmapIntermediate[ngrid*i  +jp1][k+2*l];
          f[2*k+1][2*l+0]=cmapIntermediate[ngrid*ip1+j  ][k+2*l];
          f[2*k+1][2*l+1]=cmapIntermediate[ngrid*ip1+jp1][k+2*l];
        }
      }
      for (k=0; k<4; k++) {
        for (l=0; l<4; l++) {
          P[k][l]=0;
          for (m=0; m<4; m++) {
            // P=f*T'
            P[k][l]+=f[k][m]*T[l][m]; // T transpose
          }
        }
      }
      for (k=0; k<4; k++) {
        for (l=0; l<4; l++) {
          cmapParameters[ngrid*i+j][k][l]=0;
          for (m=0; m<4; m++) {
            // a=T*P
            cmapParameters[ngrid*i+j][k][l]+=T[k][m]*P[m][l];
          }
        }
      }
    }
  }
}

real (*alloc_kcmapPtr(int ngrid,real scaling,real *kcmap))[4][4]
{
  real (*cmapIntermediate)[4]; // E dEdphi dEdpsi ddEdphidpsi
  real (*cmapParameters)[4][4]; // E_(phi_i,psi_j)
  real (*kcmapPtr)[4][4];
  real y[ngrid]; // target for cubic spline interpolation
  real dy[ngrid]; // derivative of target for cubid spline interpolation
  int i,j;

  cmapIntermediate=(real(*)[4])calloc(ngrid*ngrid,sizeof(real[4]));;
  cmapParameters=(real(*)[4][4])calloc(ngrid*ngrid,sizeof(real[4][4]));;
  cudaMalloc(&kcmapPtr,ngrid*ngrid*sizeof(real[4][4]));

  // 0 element of intermediate is the function values
  for (i=0; i<ngrid; i++) {
    for (j=0; j<ngrid; j++) {
      cmapIntermediate[ngrid*i+j][0]=scaling*kcmap[ngrid*i+j];
    }
  }

  // 1 element of intermediate is gradient with respect to phi
  for (j=0; j<ngrid; j++) {
    for (i=0; i<ngrid; i++) {
      y[i]=scaling*kcmap[ngrid*i+j];
    }
    cmap_cubic_spline_setup(ngrid,y,dy);
    for (i=0; i<ngrid; i++) {
      cmapIntermediate[ngrid*i+j][1]=dy[i];
    }
  }

  // 2 element of intermediate is gradient with respect to psi
  for (i=0; i<ngrid; i++) {
    for (j=0; j<ngrid; j++) {
      y[j]=scaling*kcmap[ngrid*i+j];
    }
    cmap_cubic_spline_setup(ngrid,y,dy);
    for (j=0; j<ngrid; j++) {
      cmapIntermediate[ngrid*i+j][2]=dy[j];
    }
  }

  // 3 element of intermediate is gradient with respect to both
  for (i=0; i<ngrid; i++) {
    for (j=0; j<ngrid; j++) {
      y[j]=cmapIntermediate[ngrid*i+j][1]; // targets y are derivatives with respect to phi
    }
    cmap_cubic_spline_setup(ngrid,y,dy);
    for (j=0; j<ngrid; j++) {
      cmapIntermediate[ngrid*i+j][3]=dy[j];
    }
  }

  // Make 16 bicubic parameters at each point from the slopes determined with the splines.
  bicubic_setup(ngrid,cmapIntermediate,cmapParameters);

  cudaMemcpy(kcmapPtr,cmapParameters,ngrid*ngrid*sizeof(real[4][4]),cudaMemcpyHostToDevice);

  if (cmapIntermediate) free(cmapIntermediate);
  if (cmapParameters) free(cmapParameters);

  return kcmapPtr;
}

void Potential::initialize(System *system)
{
  int i,j,k,l;
  Parameters *param=system->parameters;
  Structure *struc=system->structure;
  Msld *msld=system->msld;
  bool soft;

  bonds_tmp.clear();
  angles_tmp.clear();
  dihes_tmp.clear();
  imprs_tmp.clear();
  cmaps_tmp.clear();
  softBonds_tmp.clear();
  softAngles_tmp.clear();
  softDihes_tmp.clear();
  softImprs_tmp.clear();
  softCmaps_tmp.clear();
  cons_tmp.clear();

  for (i=0; i<struc->bondList.size(); i++) {
    TypeName2 type;
    struct BondPotential bond;
    struct BondParameter bp;
    // Get participating atoms
    for (j=0; j<2; j++) {
      bond.idx[j]=struc->bondList[i].i[j];
      type.t[j]=struc->atomList[bond.idx[j]].atomTypeName;
    }
    if (struc->shakeHbond && (type.t[0][0]=='H' || type.t[1][0]=='H')) {
      // Add to constraints
      for (j=0; j<2; j++) {
        cons_tmp[bond.idx[j]].insert(bond.idx[1-j]);
      }
    } else {
      // Add to bonds
      // Get their MSLD scaling
      soft=msld->bond_scaling(bond.idx,bond.siteBlock);
      // Get their parameters
      if (param->bondParameter.count(type)==0) {
        fatal(__FILE__,__LINE__,"No bond parameter for %6d(%6s) %6d(%6s)\n",bond.idx[0],type.t[0].c_str(),bond.idx[1],type.t[1].c_str());
      }
      bp=param->bondParameter[type];
      bond.kb=bp.kb;
      bond.b0=bp.b0;
      // Separate out the constrained bonds
      if (system->structure->shakeHbond && (type.t[0][0]=='H' || type.t[1][0]=='H')) {
        fatal(__FILE__,__LINE__,"hbond constraints not yet implemented (NYI)\n");
      } else if (soft) {
        softBonds_tmp.push_back(bond);
      } else {
        bonds_tmp.push_back(bond);
      }
    }
  }
  bond12Count=bonds_tmp.size();
  softBond12Count=softBonds_tmp.size();

  for (i=0; i<struc->angleList.size(); i++) {
    TypeName3 type;
    struct BondPotential bond; // urey bradley
    struct AnglePotential angle;
    struct AngleParameter ap;
    bool softU;
    // Get participating atoms
    for (j=0; j<3; j++) {
      angle.idx[j]=struc->angleList[i].i[j];
      type.t[j]=struc->atomList[angle.idx[j]].atomTypeName;
    }
    bond.idx[0]=angle.idx[0];
    bond.idx[1]=angle.idx[2];
    // Get their MSLD scaling
    soft=msld->angle_scaling(angle.idx,angle.siteBlock);
    softU=msld->ureyb_scaling(angle.idx,bond.siteBlock);
    // Get their parameters
    if (param->angleParameter.count(type)==0) {
      fatal(__FILE__,__LINE__,"No angle parameter for %6d(%6s) %6d(%6s) %6d(%6s)\n",angle.idx[0],type.t[0].c_str(),angle.idx[1],type.t[1].c_str(),angle.idx[2],type.t[2].c_str());
    }
    ap=param->angleParameter[type];
    angle.kangle=ap.kangle;
    angle.angle0=ap.angle0;
    bond.kb=ap.kureyb;
    bond.b0=ap.ureyb0;
    if (soft) {
      softAngles_tmp.push_back(angle);
    } else {
      angles_tmp.push_back(angle);
    }
    // Only include urey bradley if it's nonzero
    if (bond.kb != 0) {
      if (softU) {
        softBonds_tmp.push_back(bond);
      } else {
        bonds_tmp.push_back(bond);
      }
    }
  }
  bond13Count=bonds_tmp.size()-bond12Count;
  softBond13Count=softBonds_tmp.size()-softBond12Count;

  for (i=0; i<struc->diheList.size(); i++) {
    TypeName4 type,typx;
    struct DihePotential dihe;
    std::vector<struct DiheParameter> dp;
    // Get participating atoms
    for (j=0; j<4; j++) {
      dihe.idx[j]=struc->diheList[i].i[j];
      type.t[j]=struc->atomList[dihe.idx[j]].atomTypeName;
    }
    // Get their MSLD scaling
    soft=msld->dihe_scaling(dihe.idx,dihe.siteBlock);
    // Get their parameters
    if (param->diheParameter.count(type)==1) {
      dp=param->diheParameter[type];
    } else {
      typx=type;
      typx.t[0]="X";
      typx.t[3]="X";
      if (param->diheParameter.count(typx)==1) {
        dp=param->diheParameter[typx];
      } else {
        fatal(__FILE__,__LINE__,"No dihe parameter for %6d(%6s) %6d(%6s) %6d(%6s) %6d(%6s)\n",dihe.idx[0],type.t[0].c_str(),dihe.idx[1],type.t[1].c_str(),dihe.idx[2],type.t[2].c_str(),dihe.idx[3],type.t[3].c_str());
      }
    }
    bool rest=false;
    if (system->msld->rest) {
      for (j=0; j<4; j++) {
        rest+=system->msld->rest[dihe.idx[j]];
      }
    }
    // Include each harmonic as a separate dihedral.
    for (j=0; j<dp.size(); j++) {
      dihe.kdih=dp[j].kdih;
      dihe.ndih=dp[j].ndih;
      dihe.dih0=dp[j].dih0;
      if (rest) dihe.kdih*=system->msld->restScaling;
      if (soft) {
        softDihes_tmp.push_back(dihe);
      } else {
        dihes_tmp.push_back(dihe);
      }
    }
  }

  for (i=0; i<struc->imprList.size(); i++) {
    TypeName4 type,typx;
    struct ImprPotential impr;
    struct ImprParameter ip;
    // Get participating atoms
    for (j=0; j<4; j++) {
      impr.idx[j]=struc->imprList[i].i[j];
      type.t[j]=struc->atomList[impr.idx[j]].atomTypeName;
    }
    // Get their MSLD scaling
    soft=msld->impr_scaling(impr.idx,impr.siteBlock);
    // Get their parameters
    if (param->imprParameter.count(type)==1) { // 1st ABCD
      ip=param->imprParameter[type];
    } else {
      typx=type;
      typx.t[1]="X";
      typx.t[2]="X";
      if (param->imprParameter.count(typx)==1) { // 2nd AXXD
        ip=param->imprParameter[typx];
      } else {
        typx=type;
        typx.t[0]="X";
        if (param->imprParameter.count(typx)==1) { // 3rd XBCD
          ip=param->imprParameter[typx];
        } else {
          typx.t[3]="X";
          if (param->imprParameter.count(typx)==1) { // 4th XBCX
            ip=param->imprParameter[typx];
          } else {
            typx=type;
            typx.t[0]="X";
            typx.t[1]="X";
            if (param->imprParameter.count(typx)==1) { // 5th AXXD
              ip=param->imprParameter[typx];
            } else {
              fatal(__FILE__,__LINE__,"No impr parameter for %6d(%6s) %6d(%6s) %6d(%6s) %6d(%6s)\n",impr.idx[0],type.t[0].c_str(),impr.idx[1],type.t[1].c_str(),impr.idx[2],type.t[2].c_str(),impr.idx[3],type.t[3].c_str());
            }
          }
        }
      }
    }
    bool rest=false;
    if (system->msld->rest) {
      for (j=0; j<4; j++) {
        rest+=system->msld->rest[impr.idx[j]];
      }
    }
    impr.kimp=ip.kimp;
    impr.nimp=ip.nimp;
    impr.imp0=ip.imp0;
    if (rest) impr.kimp*=system->msld->restScaling;
    if (soft) {
      softImprs_tmp.push_back(impr);
    } else {
      imprs_tmp.push_back(impr);
    }
  }

  for (i=0; i<struc->cmapList.size(); i++) {
    TypeName8O type;
    struct CmapPotential cmap;
    struct CmapParameter cp;
    // Get participating atoms
    for (j=0; j<8; j++) {
      cmap.idx[j]=struc->cmapList[i].i[j];
      type.t[j]=struc->atomList[cmap.idx[j]].atomTypeName;
    }
    // Get their MSLD scaling
    soft=msld->cmap_scaling(cmap.idx,cmap.siteBlock);
    // Get their parameters
    if (param->cmapParameter.count(type)==1) {
      cp=param->cmapParameter[type];
    } else {
      fatal(__FILE__,__LINE__,"No cmap parameter for\n%6d(%6s) %6d(%6s) %6d(%6s) %6d(%6s)\n%6d(%6s) %6d(%6s) %6d(%6s) %6d(%6s)\n",cmap.idx[0],type.t[0].c_str(),cmap.idx[1],type.t[1].c_str(),cmap.idx[2],type.t[2].c_str(),cmap.idx[3],type.t[3].c_str(),cmap.idx[4],type.t[4].c_str(),cmap.idx[5],type.t[5].c_str(),cmap.idx[6],type.t[6].c_str(),cmap.idx[7],type.t[7].c_str());
    }
    bool rest=false;
    if (system->msld->rest) {
      for (j=0; j<8; j++) {
        rest+=system->msld->rest[cmap.idx[j]];
      }
    }
    cmap.ngrid=cp.ngrid;
    // See if we have to copy the data onto the GPU or if it's already there
    if (rest) {
      if (cmapRestTypeToPtr.count(type)!=1) {
        cmapRestTypeToPtr[type]=alloc_kcmapPtr(cp.ngrid,system->msld->restScaling,cp.kcmap);
      }
      cmap.kcmapPtr=cmapRestTypeToPtr[type];
    } else {
      if (cmapTypeToPtr.count(type)!=1) {
        cmapTypeToPtr[type]=alloc_kcmapPtr(cp.ngrid,1.0,cp.kcmap);
      }
      cmap.kcmapPtr=cmapTypeToPtr[type];
    }
    if (soft) {
      softCmaps_tmp.push_back(cmap);
    } else {
      cmaps_tmp.push_back(cmap);
    }
  }

  // ---------- Bonded setup complete, copy over data ----------
  bondCount=bonds_tmp.size();
  bonds=(struct BondPotential*)calloc(bondCount,sizeof(struct BondPotential));
  cudaMalloc(&(bonds_d),bondCount*sizeof(struct BondPotential));
  for (i=0; i<bondCount; i++) {
    bonds[i]=bonds_tmp[i];
  }
  cudaMemcpy(bonds_d,bonds,bondCount*sizeof(struct BondPotential),cudaMemcpyHostToDevice);

  angleCount=angles_tmp.size();
  angles=(struct AnglePotential*)calloc(angleCount,sizeof(struct AnglePotential));
  cudaMalloc(&(angles_d),angleCount*sizeof(struct AnglePotential));
  for (i=0; i<angleCount; i++) {
    angles[i]=angles_tmp[i];
  }
  cudaMemcpy(angles_d,angles,angleCount*sizeof(struct AnglePotential),cudaMemcpyHostToDevice);

  diheCount=dihes_tmp.size();
  dihes=(struct DihePotential*)calloc(diheCount,sizeof(struct DihePotential));
  cudaMalloc(&(dihes_d),diheCount*sizeof(struct DihePotential));
  for (i=0; i<diheCount; i++) {
    dihes[i]=dihes_tmp[i];
  }
  cudaMemcpy(dihes_d,dihes,diheCount*sizeof(struct DihePotential),cudaMemcpyHostToDevice);

  imprCount=imprs_tmp.size();
  imprs=(struct ImprPotential*)calloc(imprCount,sizeof(struct ImprPotential));
  cudaMalloc(&(imprs_d),imprCount*sizeof(struct ImprPotential));
  for (i=0; i<imprCount; i++) {
    imprs[i]=imprs_tmp[i];
  }
  cudaMemcpy(imprs_d,imprs,imprCount*sizeof(struct ImprPotential),cudaMemcpyHostToDevice);

  cmapCount=cmaps_tmp.size();
  cmaps=(struct CmapPotential*)calloc(cmapCount,sizeof(struct CmapPotential));
  cudaMalloc(&(cmaps_d),cmapCount*sizeof(struct CmapPotential));
  for (i=0; i<cmapCount; i++) {
    cmaps[i]=cmaps_tmp[i];
  }
  cudaMemcpy(cmaps_d,cmaps,cmapCount*sizeof(struct CmapPotential),cudaMemcpyHostToDevice);
  // soft bonds now
  softBondCount=softBonds_tmp.size();
  softBonds=(struct BondPotential*)calloc(softBondCount,sizeof(struct BondPotential));
  cudaMalloc(&(softBonds_d),softBondCount*sizeof(struct BondPotential));
  for (i=0; i<softBondCount; i++) {
    softBonds[i]=softBonds_tmp[i];
  }
  cudaMemcpy(softBonds_d,softBonds,softBondCount*sizeof(struct BondPotential),cudaMemcpyHostToDevice);

  softAngleCount=softAngles_tmp.size();
  softAngles=(struct AnglePotential*)calloc(softAngleCount,sizeof(struct AnglePotential));
  cudaMalloc(&(softAngles_d),softAngleCount*sizeof(struct AnglePotential));
  for (i=0; i<softAngleCount; i++) {
    softAngles[i]=softAngles_tmp[i];
  }
  cudaMemcpy(softAngles_d,softAngles,softAngleCount*sizeof(struct AnglePotential),cudaMemcpyHostToDevice);

  softDiheCount=softDihes_tmp.size();
  softDihes=(struct DihePotential*)calloc(softDiheCount,sizeof(struct DihePotential));
  cudaMalloc(&(softDihes_d),softDiheCount*sizeof(struct DihePotential));
  for (i=0; i<softDiheCount; i++) {
    softDihes[i]=softDihes_tmp[i];
  }
  cudaMemcpy(softDihes_d,softDihes,softDiheCount*sizeof(struct DihePotential),cudaMemcpyHostToDevice);

  softImprCount=softImprs_tmp.size();
  softImprs=(struct ImprPotential*)calloc(softImprCount,sizeof(struct ImprPotential));
  cudaMalloc(&(softImprs_d),softImprCount*sizeof(struct ImprPotential));
  for (i=0; i<softImprCount; i++) {
    softImprs[i]=softImprs_tmp[i];
  }
  cudaMemcpy(softImprs_d,softImprs,softImprCount*sizeof(struct ImprPotential),cudaMemcpyHostToDevice);

  softCmapCount=softCmaps_tmp.size();
  softCmaps=(struct CmapPotential*)calloc(softCmapCount,sizeof(struct CmapPotential));
  cudaMalloc(&(softCmaps_d),softCmapCount*sizeof(struct CmapPotential));
  for (i=0; i<softCmapCount; i++) {
    softCmaps[i]=softCmaps_tmp[i];
  }
  cudaMemcpy(softCmaps_d,softCmaps,softCmapCount*sizeof(struct CmapPotential),cudaMemcpyHostToDevice);
  // ---------- Data copy complete ----------



  atomCount=struc->atomList.size();
  charge=(real*)calloc(atomCount,sizeof(real));
  cudaMalloc(&charge_d,atomCount*sizeof(real));
  for (i=0; i<atomCount; i++) {
    charge[i]=struc->atomList[i].charge;
    if (system->msld->rest) {
      if (system->msld->rest[i]) {
        charge[i]*=sqrt(system->msld->restScaling);
      }
    }
  }
  cudaMemcpy(charge_d,charge,atomCount*sizeof(real),cudaMemcpyHostToDevice);

  // Set up exclusions
  virtExcl=new std::set<int>[atomCount];
  bondExcl=new std::set<int>[atomCount];
  angleExcl=new std::set<int>[atomCount];
  diheExcl=new std::set<int>[atomCount];
  msldExcl=new std::set<int>[atomCount];
  allExcl=new std::set<int>[atomCount];

  // replaced struc->bondList with bondList_tmp
  std::vector<struct Int2> bondList_tmp;
  bondList_tmp.clear();
  for (i=0; i<struc->bondList.size(); i++) {
    Int2 ij=struc->bondList[i];
    bondList_tmp.push_back(ij);
  }
  // Set virtExcl
  for (i=0; i<struc->virt2List.size(); i++) {
    Int2 ij;
    ij.i[0]=struc->virt2List[i].vidx;
    ij.i[1]=struc->virt2List[i].hidx[0];
    for (j=0; j<2; j++) {
      if (allExcl[ij.i[j]].count(ij.i[1-j]) == 0 && system->msld->interacting(ij.i[0],ij.i[1])) {
        virtExcl[ij.i[j]].insert(ij.i[1-j]);
        bondExcl[ij.i[j]].insert(ij.i[1-j]);
        allExcl[ij.i[j]].insert(ij.i[1-j]);
      }
    }
  }
  for (i=0; i<struc->virt3List.size(); i++) {
    Int2 ij;
    ij.i[0]=struc->virt3List[i].vidx;
    ij.i[1]=struc->virt3List[i].hidx[0];
    for (j=0; j<2; j++) {
      if (allExcl[ij.i[j]].count(ij.i[1-j]) == 0 && system->msld->interacting(ij.i[0],ij.i[1])) {
        virtExcl[ij.i[j]].insert(ij.i[1-j]);
        bondExcl[ij.i[j]].insert(ij.i[1-j]);
        allExcl[ij.i[j]].insert(ij.i[1-j]);
      }
    }
  }
  // Set bondExcl
  for (i=0; i<bondList_tmp.size(); i++) {
    Int2 ij=bondList_tmp[i];
    for (j=0; j<2; j++) {
      // Put in bonds first
      if (allExcl[ij.i[j]].count(ij.i[1-j]) == 0 && system->msld->interacting(ij.i[0],ij.i[1])) {
        bondExcl[ij.i[j]].insert(ij.i[1-j]);
        allExcl[ij.i[j]].insert(ij.i[1-j]);
      }
      // Then copy any exclusions for virtual sites too
      for (std::set<int>::iterator kk=virtExcl[ij.i[1-j]].begin(); kk!=virtExcl[ij.i[1-j]].end(); kk++) {
        Int2 jk;
        jk.i[0]=ij.i[j];
        jk.i[1]=*kk;
        if (jk.i[0]!=jk.i[1]) {
          for (l=0; l<2; l++) {
            if (allExcl[jk.i[l]].count(jk.i[1-l]) == 0 && system->msld->interacting(jk.i[0],jk.i[1])) {
              bondExcl[jk.i[l]].insert(jk.i[1-l]);
              allExcl[jk.i[l]].insert(jk.i[1-l]);
            }
          }
        }
      }
    }
  }
  // Set angleExcl
  for (i=0; i<bondList_tmp.size(); i++) {
    Int2 ij=bondList_tmp[i];
    for (j=0; j<2; j++) {
      for (std::set<int>::iterator kk=bondExcl[ij.i[1-j]].begin(); kk!=bondExcl[ij.i[1-j]].end(); kk++) {
        Int2 jk;
        jk.i[0]=ij.i[j];
        jk.i[1]=*kk;
        if (jk.i[0]!=jk.i[1]) {
          for (l=0; l<2; l++) {
            if (allExcl[jk.i[l]].count(jk.i[1-l]) == 0 && system->msld->interacting(jk.i[0],jk.i[1])) {
              angleExcl[jk.i[l]].insert(jk.i[1-l]);
              allExcl[jk.i[l]].insert(jk.i[1-l]);
            }
          }
        }
      }
    }
  }
  // Set diheExcl
  for (i=0; i<bondList_tmp.size(); i++) {
    Int2 ij=bondList_tmp[i];
    for (j=0; j<2; j++) {
      for (std::set<int>::iterator kk=angleExcl[ij.i[1-j]].begin(); kk!=angleExcl[ij.i[1-j]].end(); kk++) {
        Int2 jk;
        jk.i[0]=ij.i[j];
        jk.i[1]=*kk;
        if (jk.i[0]!=jk.i[1]) {
          for (l=0; l<2; l++) {
            if (allExcl[jk.i[l]].count(jk.i[1-l]) == 0 && system->msld->interacting(jk.i[0],jk.i[1])) {
              diheExcl[jk.i[l]].insert(jk.i[1-l]);
              allExcl[jk.i[l]].insert(jk.i[1-l]);
            }
          }
        }
      }
    }
  }
  // Set msldExcl
  // #warning "This is an N^2 algorithm, need to set up better data structures in Msld class to make it N"
  /*for (i=0; i<atomCount; i++) {
    for (j=0; j<atomCount; j++) {
      if (!system->msld->interacting(i,j)) {
        msldExcl[i].insert(j);
        msldExcl[j].insert(i);
        allExcl[i].insert(j);
        allExcl[j].insert(i);
      }
    }
  }*/
  // This should no longer be N^2.
  for (i=0; i<system->msld->siteCount; i++) {
    for (j=system->msld->siteBound[i]; j<system->msld->siteBound[i+1]; j++) {
      for (k=j+1; k<system->msld->siteBound[i+1]; k++) {
        for (std::set<int>::iterator ii=system->msld->atomsByBlock[j].begin(); ii!=msld->atomsByBlock[j].end(); ii++) {
          for (std::set<int>::iterator jj=system->msld->atomsByBlock[k].begin(); jj!=msld->atomsByBlock[k].end(); jj++) {
            if (system->msld->interacting(*ii,*jj)) fatal(__FILE__,__LINE__,"Atoms %d and %d should not be in exclusions\n",*ii,*jj);
            msldExcl[*ii].insert(*jj);
            msldExcl[*jj].insert(*ii);
            allExcl[*ii].insert(*jj);
            allExcl[*jj].insert(*ii);
          }
        }
      }
    }
  }

  for (i=0; i<atomCount; i++) {
    for (std::set<int>::iterator jj=diheExcl[i].begin(); jj!=diheExcl[i].end(); jj++) {
      if (i<*jj) {
        TypeName2 type;
        Nb14Potential nb14;
        struct NbondParameter np;
        // Get participating atoms
        nb14.idx[0]=i;
        nb14.idx[1]=*jj;
        for (k=0; k<2; k++) {
          type.t[k]=struc->atomList[nb14.idx[k]].atomTypeName;
        }
        // Get their MSLD scaling
        msld->nb14_scaling(nb14.idx,nb14.siteBlock);
        // Get their parameters
        nb14.qxq=charge[nb14.idx[0]]*charge[nb14.idx[1]];
        if (param->nbfixParameter.count(type)==1) {
          np=param->nbfixParameter[type];
        } else {
          struct NbondParameter npij[2];
          for (k=0; k<2; k++) {
            if (param->nbondParameter.count(type.t[k])==1) {
              npij[k]=param->nbondParameter[type.t[k]];
            } else {
              fatal(__FILE__,__LINE__,"Nonbonded parameter for atom %d type %s not found\n",nb14.idx[k],type.t[k].c_str());
            }
          }
          np.eps14=sqrt(npij[0].eps14*npij[1].eps14);
          np.sig14=npij[0].sig14+npij[1].sig14;
        }
        if (system->msld->rest) {
          for (j=0; j<2; j++) {
            if (system->msld->rest[nb14.idx[j]]) {
              np.eps14*=sqrt(system->msld->restScaling);
            }
          }
        }
        real sig6=np.sig14*np.sig14;
        sig6*=(sig6*sig6);
        nb14.c12=np.eps14*sig6*sig6;
        nb14.c6=2*np.eps14*sig6;
        nb14s_tmp.push_back(nb14);
      }
    }
  }

  if (system->run->usePME) {

  for (i=0; i<atomCount; i++) {
    for (std::set<int>::iterator jj=allExcl[i].begin(); jj!=allExcl[i].end(); jj++) {
      if (i<*jj && diheExcl[i].count(*jj)==0) {
        NbExPotential nbex;
        // Get participating atoms
        nbex.idx[0]=i;
        nbex.idx[1]=*jj;
        // Get their MSLD scaling
        if (msld->nbex_scaling(nbex.idx,nbex.siteBlock)) {
          // Get their parameters
          nbex.qxq=charge[nbex.idx[0]]*charge[nbex.idx[1]];
          nbexs_tmp.push_back(nbex);
        }
      }
    }
  }

  }

  for (i=0; i<atomCount; i++) {
    for (std::set<int>::iterator jj=allExcl[i].begin(); jj!=allExcl[i].end(); jj++) {
      ExclPotential excl;
      excl.idx[0]=i;
      excl.idx[1]=*jj;
      excls_tmp.push_back(excl);
    }
  }

  nb14Count=nb14s_tmp.size();
  nb14s=(struct Nb14Potential*)calloc(nb14Count,sizeof(struct Nb14Potential));
  cudaMalloc(&(nb14s_d),nb14Count*sizeof(struct Nb14Potential));
  for (i=0; i<nb14Count; i++) {
    nb14s[i]=nb14s_tmp[i];
  }
  cudaMemcpy(nb14s_d,nb14s,nb14Count*sizeof(struct Nb14Potential),cudaMemcpyHostToDevice);

  if (system->run->usePME) {

  nbexCount=nbexs_tmp.size();
  nbexs=(struct NbExPotential*)calloc(nbexCount,sizeof(struct NbExPotential));
  cudaMalloc(&(nbexs_d),nbexCount*sizeof(struct NbExPotential));
  for (i=0; i<nbexCount; i++) {
    nbexs[i]=nbexs_tmp[i];
  }
  cudaMemcpy(nbexs_d,nbexs,nbexCount*sizeof(struct NbExPotential),cudaMemcpyHostToDevice);

  }

  exclCount=excls_tmp.size();
  excls=(struct ExclPotential*)calloc(exclCount,sizeof(struct ExclPotential));
  cudaMalloc(&(excls_d),exclCount*sizeof(struct ExclPotential));
  for (i=0; i<exclCount; i++) {
    excls[i]=excls_tmp[i];
  }
  cudaMemcpy(excls_d,excls,exclCount*sizeof(struct ExclPotential),cudaMemcpyHostToDevice);

  if (system->run->usePME) {

  // Choose PME grid sizes
  int goodSizes[]={32,27,24,20,18,16};
  real boxtmp[3]={(real)(system->state->box.a.x),(real)(system->state->box.a.y),(real)(system->state->box.a.z)};
  for (i=0; i<3; i++) {
    if (system->run->gridSpace>0) {
      real minDim=boxtmp[i]/system->run->gridSpace;
      for (j=1; minDim>=32*j; j*=2) {
        ;
      }
      for (k=0; k<5 && minDim<j*goodSizes[k]; k++) { // guaranteed to pass k=0
        ;
      }
      gridDimPME[i]=j*goodSizes[k-1];
    } else {
      gridDimPME[i]=system->run->grid[i];
    }
    fprintf(stdout,"PME grid(%d) size: %d\n",i,gridDimPME[i]);
  }

  cudaMalloc(&chargeGridPME_d,gridDimPME[0]*gridDimPME[1]*gridDimPME[2]*sizeof(myCufftReal));
  cudaMalloc(&fourierGridPME_d,gridDimPME[0]*gridDimPME[1]*(gridDimPME[2]/2+1)*sizeof(myCufftComplex));
  cudaMalloc(&potentialGridPME_d,gridDimPME[0]*gridDimPME[1]*gridDimPME[2]*sizeof(myCufftReal));
#ifdef USE_TEXTURE
  {
    cudaResourceDesc resDesc;
    memset(&resDesc,0,sizeof(resDesc));
    resDesc.resType=cudaResourceTypeLinear;
    resDesc.res.linear.devPtr=potentialGridPME_d;
    resDesc.res.linear.desc=cudaCreateChannelDesc<real>();
    resDesc.res.linear.sizeInBytes=gridDimPME[0]*gridDimPME[1]*gridDimPME[2]*sizeof(real);
    cudaTextureDesc texDesc;
    memset(&texDesc,0,sizeof(texDesc));
    texDesc.readMode=cudaReadModeElementType;
    cudaCreateTextureObject(&potentialGridPME_tex,&resDesc,&texDesc,NULL);
  }
#endif

  bGridPME=(real*)calloc(gridDimPME[0]*gridDimPME[1]*(gridDimPME[2]/2+1),sizeof(real));
  cudaMalloc(&bGridPME_d,gridDimPME[0]*gridDimPME[1]*(gridDimPME[2]/2+1)*sizeof(real));
  int order=system->run->orderEwald;
  // Only have to support orders 4, 6, and 8
  real Meven[9]={0,1,0,0,0,0,0,0,0};
  real Modd[9]={0,0,0,0,0,0,0,0,0};
  for (i=2; i<order; i+=2) {
    for (l=0; l<i+1; l++) {
      Modd[l]=l*Meven[l]/i+(i+1-l)*Meven[i+1-l]/i;
    }
    for (l=0; l<i+2; l++) {
      Meven[l]=l*Modd[l]/(i+1)+(i+2-l)*Modd[i+2-l]/(i+1);
    }
  }
  real *invbx2=(real*)calloc(gridDimPME[0],sizeof(real));
  for (i=0; i<gridDimPME[0]; i++) {
    myCufftComplex bx;
    bx.x=0;
    bx.y=0;
    for (l=1; l<order; l++) {
      bx.x+=Meven[l]*cos((2*M_PI*i*l)/gridDimPME[0]);
      bx.y+=Meven[l]*sin((2*M_PI*i*l)/gridDimPME[0]);
    }
    invbx2[i]=1.0/(bx.x*bx.x+bx.y*bx.y);
  }
  real *invby2=(real*)calloc(gridDimPME[1],sizeof(real));
  for (i=0; i<gridDimPME[1]; i++) {
    myCufftComplex by;
    by.x=0;
    by.y=0;
    for (l=1; l<order; l++) {
      by.x+=Meven[l]*cos((2*M_PI*i*l)/gridDimPME[1]);
      by.y+=Meven[l]*sin((2*M_PI*i*l)/gridDimPME[1]);
    }
    invby2[i]=1.0/(by.x*by.x+by.y*by.y);
  }
  real *invbz2=(real*)calloc(gridDimPME[2]/2+1,sizeof(real));
  for (i=0; i<(gridDimPME[2]/2+1); i++) {
    myCufftComplex bz;
    bz.x=0;
    bz.y=0;
    for (l=1; l<order; l++) {
      bz.x+=Meven[l]*cos((2*M_PI*i*l)/gridDimPME[2]);
      bz.y+=Meven[l]*sin((2*M_PI*i*l)/gridDimPME[2]);
    }
    invbz2[i]=1.0/(bz.x*bz.x+bz.y*bz.y);
  }
  for (i=0; i<gridDimPME[0]; i++) {
    for (j=0; j<gridDimPME[1]; j++) {
      for (k=0; k<(gridDimPME[2]/2+1); k++) {
        bGridPME[(i*gridDimPME[1]+j)*(gridDimPME[2]/2+1)+k]=invbx2[i]*invby2[j]*invbz2[k];
      }
    }
  }
  free(invbx2);
  free(invby2);
  free(invbz2);
  cudaMemcpy(bGridPME_d,bGridPME,gridDimPME[0]*gridDimPME[1]*(gridDimPME[2]/2+1)*sizeof(real),cudaMemcpyHostToDevice);

  cufftCreate(&planFFTPME);
  cufftMakePlan3d(planFFTPME,gridDimPME[0],gridDimPME[1],gridDimPME[2],MYCUFFT_R2C,&bufferSizeFFTPME);
  cufftSetStream(planFFTPME,system->run->nbrecipStream);
  cufftCreate(&planIFFTPME);
  cufftMakePlan3d(planIFFTPME,gridDimPME[0],gridDimPME[1],gridDimPME[2],MYCUFFT_C2R,&bufferSizeIFFTPME);
  cufftSetStream(planIFFTPME,system->run->nbrecipStream);

  }

  // Count each nonbonded type
  typeCount.clear();
  for (i=0; i<atomCount; i++) {
    std::string type=struc->atomList[i].atomTypeName;
    if (system->msld->rest) {
      if (system->msld->rest[i]) {
        type="REST_"+type;
      }
    }
    if (typeCount.count(type)==1) {
      typeCount[type]++;
    } else {
      typeCount[type]=1;
    }
  }
  // Sort by counts
  typeSort.clear();
  for (std::map<std::string,int>::iterator ii=typeCount.begin(); ii!=typeCount.end(); ii++) {
    struct CountType entry;
    entry.count=ii->second;
    entry.type=ii->first;
    typeSort.insert(entry);
  }
  // Get sorted lists of types
  i=0;
  typeList.clear(); // number to string
  typeLookup.clear(); // string to number
  for (std::set<struct CountType>::iterator ii=typeSort.begin(); ii!=typeSort.end(); ii++) {
    typeList.push_back(ii->type);
    typeLookup[ii->type]=i;
    i++;
  }
  // Save nonbonded information
  nbonds=(NbondPotential*)calloc(atomCount,sizeof(NbondPotential));
  cudaMalloc(&nbonds_d,atomCount*sizeof(NbondPotential));
  for (i=0; i<atomCount; i++) {
    struct NbondPotential nbond;
    std::string type;
    type=struc->atomList[i].atomTypeName;
    if (system->msld->rest) {
      if (system->msld->rest[i]) {
        type="REST_"+type;
      }
    }
    // Get their MSLD scaling
    msld->nbond_scaling(&i,&nbond.siteBlock);
    // Get their parameters
    nbond.q=charge[i];
    nbond.typeIdx=typeLookup[type];
    nbonds[i]=nbond;
  }
  cudaMemcpy(nbonds_d,nbonds,atomCount*sizeof(NbondPotential),cudaMemcpyHostToDevice);
  // Save van der Waals interaction parameters
  vdwParameterCount=typeList.size();
  vdwParameters=(VdwPotential*)calloc(vdwParameterCount*vdwParameterCount,sizeof(VdwPotential));
  cudaMalloc(&vdwParameters_d,vdwParameterCount*vdwParameterCount*sizeof(VdwPotential));
  for (i=0; i<vdwParameterCount; i++) {
    TypeName2 type;
    struct NbondParameter np;
    real scaling[2];
    type.t[0]=typeList[i];
    scaling[0]=1;
    if (type.t[0].substr(0,5)=="REST_") {
      type.t[0]=type.t[0].substr(5,type.t[0].npos);
      scaling[0]*=sqrt(system->msld->restScaling);
    }
    for (j=0; j<vdwParameterCount; j++) {
      type.t[1]=typeList[j];
      scaling[1]=scaling[0];
      if (type.t[1].substr(0,5)=="REST_") {
        type.t[1]=type.t[1].substr(5,type.t[1].npos);
        scaling[1]*=sqrt(system->msld->restScaling);
      }
      if (param->nbfixParameter.count(type)==1) {
        np=param->nbfixParameter[type];
      } else {
        struct NbondParameter npij[2];
        for (k=0; k<2; k++) {
          if (param->nbondParameter.count(type.t[k])==1) {
            npij[k]=param->nbondParameter[type.t[k]];
          } else {
            fatal(__FILE__,__LINE__,"Nonbonded parameter for atom type %s not found\n",type.t[k].c_str());
          }
        }
        np.eps=sqrt(npij[0].eps*npij[1].eps);
        np.sig=npij[0].sig+npij[1].sig;
        }
      np.eps*=scaling[1];
      real sig6=np.sig*np.sig;
      sig6*=(sig6*sig6);
      vdwParameters[i*vdwParameterCount+j].c12=np.eps*sig6*sig6;
      vdwParameters[i*vdwParameterCount+j].c6=2*np.eps*sig6;
    }
  }
  cudaMemcpy(vdwParameters_d,vdwParameters,vdwParameterCount*vdwParameterCount*sizeof(VdwPotential),cudaMemcpyHostToDevice);
#ifdef USE_TEXTURE
  {
    cudaResourceDesc resDesc;
    memset(&resDesc,0,sizeof(resDesc));
    resDesc.resType=cudaResourceTypeLinear;
    resDesc.res.linear.devPtr=vdwParameters_d;
    resDesc.res.linear.desc=cudaCreateChannelDesc<real2>();
    resDesc.res.linear.sizeInBytes=vdwParameterCount*vdwParameterCount*sizeof(VdwPotential);
    cudaTextureDesc texDesc;
    memset(&texDesc,0,sizeof(texDesc));
    texDesc.readMode=cudaReadModeElementType;
    cudaCreateTextureObject(&vdwParameters_tex,&resDesc,&texDesc,NULL);
  }
#endif

  // Sort out these constraints
  triangleCons_tmp.clear();
  branch1Cons_tmp.clear();
  branch2Cons_tmp.clear();
  branch3Cons_tmp.clear();
  for (std::map<int,std::set<int> >::iterator ii=cons_tmp.begin(); ii!=cons_tmp.end(); ii++) {
    std::vector<int> consIdx;
    std::vector<std::string> consType;
    std::vector<bool> consH;
    bool consBad=false;
    consIdx.clear();
    consIdx.push_back(ii->first);
    for (std::set<int>::iterator jj=ii->second.begin(); jj!=ii->second.end(); jj++) {
      consIdx.push_back(*jj);
    }
    consType.clear();
    consH.clear();
    for (i=0; i<consIdx.size(); i++) {
      consType.push_back(struc->atomList[consIdx[i]].atomTypeName);
      consH.push_back(consType[i][0]=='H');
    }
    // // Print it out for debugging purposes
    // fprintf(stdout,"CONS:");
    // for (i=0; i<consIdx.size(); i++) {
    //   fprintf(stdout," %5d",consIdx[i]);
    // }
    // fprintf(stdout,"\n");
    if (consIdx.size()==3) {
      // Check for triangle
      if (cons_tmp[consIdx[1]].size()==2 && cons_tmp[consIdx[2]].size()==2) {
        if (cons_tmp[consIdx[1]].count(consIdx[2])==1 && cons_tmp[consIdx[2]].count(consIdx[1])==1) {
          if (consH[0]<consH[1] || (consH[0]==consH[1]&&consIdx[0]<consIdx[1])) {
            if (consH[0]<consH[2] || (consH[0]==consH[2]&&consIdx[0]<consIdx[2])) {
              // Found properly oriented triange
              TypeName2 type;
              struct TriangleCons tc;
              k=0;
              for (i=0; i<3; i++) {
                tc.idx[i]=consIdx[i];
                type.t[0]=consType[i];
                for (j=i+1; j<3; j++) {
                  type.t[1]=consType[j];
                  if (param->bondParameter.count(type)==0) {
                    fatal(__FILE__,__LINE__,"No bond parameter for %6d(%6s) %6d(%6s)\n",consIdx[i],type.t[0].c_str(),consIdx[j],type.t[1].c_str());
                  }
                  tc.b0[k]=param->bondParameter[type].b0;
                  k++;
                }
              }
              // // Print it out for debugging purposes
              // fprintf(stdout,"Triangle:");
              // for (i=0; i<consIdx.size(); i++) {
              //   fprintf(stdout," %5d",consIdx[i]);
              // }
              // fprintf(stdout,"\n");
              triangleCons_tmp.push_back(tc);
            } // else consIdx[2] should be in 0 position
          } // else consIdx[1] should be in 0 posiiton
        } else {
          consBad=true;
        }
      // Check for branch2
      } else if (cons_tmp[consIdx[1]].size()==1 && cons_tmp[consIdx[2]].size()==1) {
        // Found properly oriented branch2
        TypeName2 type;
        struct Branch2Cons bc;
        k=0;
        for (i=0; i<3; i++) {
          bc.idx[i]=consIdx[i];
          type.t[0]=consType[i];
          if (i==0) {
            for (j=i+1; j<3; j++) {
              type.t[1]=consType[j];
              if (param->bondParameter.count(type)==0) {
                fatal(__FILE__,__LINE__,"No bond parameter for %6d(%6s) %6d(%6s)\n",consIdx[i],type.t[0].c_str(),consIdx[j],type.t[1].c_str());
              }
              bc.b0[k]=param->bondParameter[type].b0;
              k++;
            }
          }
        }
        // // Print it out for debugging purposes
        // fprintf(stdout,"Branch2:");
        // for (i=0; i<consIdx.size(); i++) {
        //   fprintf(stdout," %5d",consIdx[i]);
        // }
        // fprintf(stdout,"\n");
        branch2Cons_tmp.push_back(bc);
      } else {
        consBad=true;
      }
    // check for branch3
    } else if (consIdx.size()==4) {
      if (cons_tmp[consIdx[1]].size()==1 && cons_tmp[consIdx[2]].size()==1 && cons_tmp[consIdx[3]].size()==1) {
        // Found properly oriented branch3
        TypeName2 type;
        struct Branch3Cons bc;
        k=0;
        for (i=0; i<4; i++) {
          bc.idx[i]=consIdx[i];
          type.t[0]=consType[i];
          if (i==0) {
            for (j=i+1; j<4; j++) {
              type.t[1]=consType[j];
              if (param->bondParameter.count(type)==0) {
                fatal(__FILE__,__LINE__,"No bond parameter for %6d(%6s) %6d(%6s)\n",consIdx[i],type.t[0].c_str(),consIdx[j],type.t[1].c_str());
              }
              bc.b0[k]=param->bondParameter[type].b0;
              k++;
            }
          }
        }
        // // Print it out for debugging purposes
        // fprintf(stdout,"Branch3:");
        // for (i=0; i<consIdx.size(); i++) {
        //   fprintf(stdout," %5d",consIdx[i]);
        // }
        // fprintf(stdout,"\n");
        branch3Cons_tmp.push_back(bc);
      } else {
        consBad=true;
      }
    // check for branch1
    } else if (consIdx.size()==2) {
      if (cons_tmp[consIdx[1]].size()==1) {
        if (consH[0]<consH[1] || (consH[0]==consH[1]&&consIdx[0]<consIdx[1])) {
          // Found properly oriented branch1
          TypeName2 type;
          struct Branch1Cons bc;
          k=0;
          for (i=0; i<2; i++) {
            bc.idx[i]=consIdx[i];
            type.t[0]=consType[i];
            if (i==0) {
              for (j=i+1; j<2; j++) {
                type.t[1]=consType[j];
                if (param->bondParameter.count(type)==0) {
                  fatal(__FILE__,__LINE__,"No bond parameter for %6d(%6s) %6d(%6s)\n",consIdx[i],type.t[0].c_str(),consIdx[j],type.t[1].c_str());
                }
                bc.b0[k]=param->bondParameter[type].b0;
                k++;
              }
            }
          }
          // // Print it out for debugging purposes
          // fprintf(stdout,"Branch1:");
          // for (i=0; i<consIdx.size(); i++) {
          //   fprintf(stdout," %5d",consIdx[i]);
          // }
          // fprintf(stdout,"\n");
          branch1Cons_tmp.push_back(bc);
        } // else consIdx[1] should be in position 0
      } // else consIdx[0] should be in position 0 because it has more partners
    } else {
      consBad=true;
    }
    if (consBad) {
      fatal(__FILE__,__LINE__,"Unrecognized constraint. %d constrained to %d atoms starting with %d\n",consIdx[0],consIdx.size(),consIdx[1]);
    }
  }

  // Triange constraints (solvent, use SETTLE)
  triangleConsCount=triangleCons_tmp.size();
  triangleCons=(struct TriangleCons*)calloc(triangleConsCount,sizeof(struct TriangleCons));
  cudaMalloc(&triangleCons_d,triangleConsCount*sizeof(struct TriangleCons));
  for (i=0; i<triangleConsCount; i++) {
    triangleCons[i]=triangleCons_tmp[i];
  }
  cudaMemcpy(triangleCons_d,triangleCons,triangleConsCount*sizeof(struct TriangleCons),cudaMemcpyHostToDevice);

  // Branch1 constraints
  branch1ConsCount=branch1Cons_tmp.size();
  branch1Cons=(struct Branch1Cons*)calloc(branch1ConsCount,sizeof(struct Branch1Cons));
  cudaMalloc(&branch1Cons_d,branch1ConsCount*sizeof(struct Branch1Cons));
  for (i=0; i<branch1ConsCount; i++) {
    branch1Cons[i]=branch1Cons_tmp[i];
  }
  cudaMemcpy(branch1Cons_d,branch1Cons,branch1ConsCount*sizeof(struct Branch1Cons),cudaMemcpyHostToDevice);

  // Branch2 constraints
  branch2ConsCount=branch2Cons_tmp.size();
  branch2Cons=(struct Branch2Cons*)calloc(branch2ConsCount,sizeof(struct Branch2Cons));
  cudaMalloc(&branch2Cons_d,branch2ConsCount*sizeof(struct Branch2Cons));
  for (i=0; i<branch2ConsCount; i++) {
    branch2Cons[i]=branch2Cons_tmp[i];
  }
  cudaMemcpy(branch2Cons_d,branch2Cons,branch2ConsCount*sizeof(struct Branch2Cons),cudaMemcpyHostToDevice);

  // Branch3 constraints
  branch3ConsCount=branch3Cons_tmp.size();
  branch3Cons=(struct Branch3Cons*)calloc(branch3ConsCount,sizeof(struct Branch3Cons));
  cudaMalloc(&branch3Cons_d,branch3ConsCount*sizeof(struct Branch3Cons));
  for (i=0; i<branch3ConsCount; i++) {
    branch3Cons[i]=branch3Cons_tmp[i];
  }
  cudaMemcpy(branch3Cons_d,branch3Cons,branch3ConsCount*sizeof(struct Branch3Cons),cudaMemcpyHostToDevice);

  // Virtual sites 2
  virtualSite2Count=system->structure->virt2List.size();
  virtualSite2=(struct VirtualSite2*)calloc(virtualSite2Count,sizeof(struct VirtualSite2));
  cudaMalloc(&virtualSite2_d,virtualSite2Count*sizeof(struct VirtualSite2));
  for (i=0; i<virtualSite2Count; i++) {
    virtualSite2[i]=system->structure->virt2List[i];
  }
  cudaMemcpy(virtualSite2_d,virtualSite2,virtualSite2Count*sizeof(struct VirtualSite2),cudaMemcpyHostToDevice);

  // Virtual sites 3
  virtualSite3Count=system->structure->virt3List.size();
  virtualSite3=(struct VirtualSite3*)calloc(virtualSite3Count,sizeof(struct VirtualSite3));
  cudaMalloc(&virtualSite3_d,virtualSite3Count*sizeof(struct VirtualSite3));
  for (i=0; i<virtualSite3Count; i++) {
    virtualSite3[i]=system->structure->virt3List[i];
  }
  cudaMemcpy(virtualSite3_d,virtualSite3,virtualSite3Count*sizeof(struct VirtualSite3),cudaMemcpyHostToDevice);

  // NOE restraints
  noeCount=system->structure->noeCount;
  noes=(struct NoePotential*)calloc(noeCount,sizeof(struct NoePotential));
  cudaMalloc(&noes_d,noeCount*sizeof(struct NoePotential));
  for (i=0; i<noeCount; i++) {
    noes[i]=system->structure->noeList[i];
  }
  cudaMemcpy(noes_d,noes,noeCount*sizeof(struct NoePotential),cudaMemcpyHostToDevice);

  // Harmonic restraints
  harmCount=system->structure->harmCount;
  harms=(struct HarmonicPotential*)calloc(harmCount,sizeof(struct HarmonicPotential));
  cudaMalloc(&harms_d,harmCount*sizeof(struct HarmonicPotential));
  real_x harmCenterNorm=0;
  harmCenter.x=0;
  harmCenter.y=0;
  harmCenter.z=0;
  for (i=0; i<harmCount; i++) {
    harms[i]=system->structure->harmList[i];
    harmCenterNorm+=harms[i].k;
    harmCenter.x+=harms[i].k*harms[i].r0.x;
    harmCenter.y+=harms[i].k*harms[i].r0.y;
    harmCenter.z+=harms[i].k*harms[i].r0.z;
  }
  if (harmCount) {
    harmCenter.x/=harmCenterNorm;
    harmCenter.y/=harmCenterNorm;
    harmCenter.z/=harmCenterNorm;
  }
  cudaMemcpy(harms_d,harms,harmCount*sizeof(struct HarmonicPotential),cudaMemcpyHostToDevice);

  // Cleaning up output coordinates
  prettifyPlan=(int(*)[2])malloc(atomCount*sizeof(int[2]));
  std::set<int> prettifyFound, prettifyMissing;
  prettifyFound.clear();
  prettifyMissing.clear();
  for (i=0; i<atomCount; i++) {
    prettifyMissing.insert(i);
  }
  for (i=0; i<atomCount; i++) {
    if (i==prettifyFound.size()) { // Get an element from prettifyMissing
      k=-1;
      j=*prettifyMissing.begin();
      prettifyPlan[prettifyFound.size()][0]=j;
      prettifyPlan[prettifyFound.size()][1]=k;
      prettifyMissing.erase(j);
      prettifyFound.insert(j);
    }
    k=prettifyPlan[i][0];
    for (std::set<int>::iterator ii=bondExcl[k].begin(); ii!=bondExcl[k].end(); ii++) { 
      j=*ii;
      if (prettifyMissing.count(j)==1) {
        prettifyPlan[prettifyFound.size()][0]=j;
        prettifyPlan[prettifyFound.size()][1]=k;
        prettifyMissing.erase(j);
        prettifyFound.insert(j);
      }
    }
  }
}

void Potential::reset_force(System *system,bool calcEnergy)
{
  cudaMemset(system->state->forceBuffer_d,0,(2*system->state->lambdaCount+3*system->state->atomCount)*sizeof(real_f));
  // #warning "Also need to do something intelligent about localForce_d size here"
  // cudaMemset(system->domdec->localForce_d,0,2*system->domdec->globalCount*sizeof(real3_f));
  // Fixed. See also localForce_d in src/domdec/domdec.cu
  cudaMemset(system->domdec->localForce_d,0,32*system->domdec->maxBlocks*sizeof(real3_f));
  if (calcEnergy) {
    cudaMemset(system->state->energy_d,0,eeend*sizeof(real_e));
  }
}

void Potential::calc_force(int step,System *system)
{
  bool calcEnergy=(step%system->run->freqNRG==0);
  int helper=(system->idCount==2); // 0 unless there are 2 GPUs, then it's 1.
  if (system->run->freqNPT>0) {
    calcEnergy=(calcEnergy||(step%system->run->freqNPT==0));
  }
#ifdef REPLICAEXCHANGE
  if (system->run->freqREx>0) {
    calcEnergy=(calcEnergy||(step%system->run->freqREx==0));
  }
#endif
  Run *r=system->run;
  State *s=system->state;

  // s->set_fd(system); // should have already been called
  reset_force(system,calcEnergy);

  cudaEventRecord(r->forceBegin,r->updateStream);

  if (system->id==helper) {
    cudaStreamWaitEvent(r->bondedStream,r->forceBegin,0);
    getforce_bond(system,calcEnergy);
    getforce_angle(system,calcEnergy);
    getforce_dihe(system,calcEnergy);
    getforce_impr(system,calcEnergy);
    getforce_cmap(system,calcEnergy);
    getforce_nb14(system,calcEnergy);
    getforce_nbex(system,calcEnergy);
    cudaEventRecord(r->bondedComplete,r->bondedStream);
    cudaStreamWaitEvent(r->updateStream,r->bondedComplete,0);
  }

  if (system->id==0) {
    cudaStreamWaitEvent(r->nbrecipStream,r->forceBegin,0);
    getforce_ewaldself(system,calcEnergy);
    getforce_ewald(system,calcEnergy);
    system->rngGPU->rand_normal(s->leapState->N,s->leapState->random,r->nbrecipStream);
    cudaEventRecord(r->nbrecipComplete,r->nbrecipStream);
    cudaStreamWaitEvent(r->updateStream,r->nbrecipComplete,0);
  }

  if (system->id==helper) {
    cudaStreamWaitEvent(r->biaspotStream,r->forceBegin,0);
    system->msld->getforce_fixedBias(system,calcEnergy);
    system->msld->getforce_variableBias(system,calcEnergy);
    system->msld->getforce_thetaBias(system,calcEnergy);
    system->msld->getforce_atomRestraints(system,calcEnergy);
    system->msld->getforce_chargeRestraints(system,calcEnergy);
    getforce_noe(system,calcEnergy);
    getforce_harm(system,calcEnergy);
    cudaEventRecord(r->biaspotComplete,r->biaspotStream);
    cudaStreamWaitEvent(r->updateStream,r->biaspotComplete,0);
  }

  if (system->domdec->id>=0) {
    cudaStreamWaitEvent(r->nbdirectStream,r->forceBegin,0);
    getforce_nbdirect(system,calcEnergy);
    cudaEventRecord(r->nbdirectComplete,r->nbdirectStream);
    cudaStreamWaitEvent(r->updateStream,r->nbdirectComplete,0);
  }

  if (system->idCount>1) {
    system->state->gather_force(system,calcEnergy);
    if (system->id==0) {
      getforce_nbdirect_reduce(system,calcEnergy);
    }
  }

  calc_virtual_force(system);

  // cudaEventRecord(r->forceComplete,r->updateStream);
}

