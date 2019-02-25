#include <string.h>
#include <cuda_runtime.h>

#include "msld/msld.h"
#include "system/system.h"
#include "io/io.h"
#include "system/selections.h"
#include "system/structure.h"
#include "system/state.h"
#include "system/potential.h"



// Class constructors
Msld::Msld() {
  blockCount=1;
  atomBlock=NULL;
  lambdaSite=NULL;
  lambda=NULL;
  lambdaForce=NULL;
  lambdaBias=NULL;
  theta=NULL;
  thetaVelocity=NULL;
  thetaMass=NULL;
  thetaInvsqrtMass=NULL;
  lambdaCharge=NULL;

  lambdaSite_d=NULL;
  lambda_d=NULL;
  lambdaForce_d=NULL;
  lambdaBias_d=NULL;
  theta_d=NULL;
  thetaVelocity_d=NULL;
  thetaForce_d=NULL;
  thetaMass_d=NULL;
  thetaInvsqrtMass_d=NULL;
  thetaRandom_d=NULL;

  blocksPerSite=NULL;
  blocksPerSite_d=NULL;
  siteBound=NULL;
  siteBound_d=NULL;

  gamma=1; // ps^-1
#warning "fnex defaults to 5.5"
  fnex=5.5;

  variableBias_tmp.clear();
  variableBias=NULL;
  variableBias_d=NULL;

  softBonds.clear();
  atomRestraints.clear();
}

Msld::~Msld() {
  if (atomBlock) free(atomBlock);
  if (lambdaSite) free(lambdaSite);
  if (lambda) free(lambda);
  if (lambdaForce) free(lambdaForce);
  if (lambdaBias) free(lambdaBias);
  if (theta) free(theta);
  if (thetaVelocity) free(thetaVelocity);
  if (thetaMass) free(thetaMass);
  if (thetaInvsqrtMass) free(thetaInvsqrtMass);
  if (lambdaCharge) free(lambdaCharge);

  if (lambdaSite_d) cudaFree(lambdaSite_d);
  if (lambda_d) cudaFree(lambda_d);
  if (lambdaForce_d) cudaFree(lambdaForce_d);
  if (lambdaBias_d) cudaFree(lambdaBias_d);
  if (theta_d) cudaFree(theta_d);
  if (thetaVelocity_d) cudaFree(thetaVelocity_d);
  if (thetaForce_d) cudaFree(thetaForce_d);
  if (thetaMass_d) cudaFree(thetaMass_d);
  if (thetaInvsqrtMass_d) cudaFree(thetaInvsqrtMass_d);
  if (thetaRandom_d) cudaFree(thetaRandom_d);

  if (blocksPerSite) free(blocksPerSite);
  if (blocksPerSite_d) cudaFree(blocksPerSite_d);
  if (siteBound) free(siteBound);
  if (siteBound_d) cudaFree(siteBound_d);
}



// Class parsing
void parse_msld(char *line,System *system)
{
  char token[MAXLENGTHSTRING];
  int i,j;

  if (system->structure==NULL) {
    fatal(__FILE__,__LINE__,"selections cannot be defined until structure has been defined\n");
  }

  if (system->msld==NULL) {
    system->msld=new Msld;
  }

  io_nexta(line,token);
  if (strcmp(token,"reset")==0) {
    if (system->msld) {
      delete(system->msld);
      system->msld=NULL;
    }
  } else if (strcmp(token,"nblocks")==0) {
    if (system->msld) {
      delete(system->msld);
    }
    system->msld=new(Msld);
    system->msld->blockCount=io_nexti(line)+1;
    system->msld->atomBlock=(int*)calloc(system->structure->atomList.size(),sizeof(int));
    system->msld->lambdaSite=(int*)calloc(system->msld->blockCount,sizeof(int));
    system->msld->lambda=(real*)calloc(system->msld->blockCount,sizeof(real));
    system->msld->lambdaForce=(real*)calloc(system->msld->blockCount,sizeof(real));
    system->msld->lambdaBias=(real*)calloc(system->msld->blockCount,sizeof(real));
    system->msld->theta=(real*)calloc(system->msld->blockCount,sizeof(real));
    system->msld->thetaVelocity=(real*)calloc(system->msld->blockCount,sizeof(real));
    system->msld->thetaMass=(real*)calloc(system->msld->blockCount,sizeof(real));
    system->msld->thetaInvsqrtMass=(real*)calloc(system->msld->blockCount,sizeof(real));
    system->msld->lambdaCharge=(real*)calloc(system->msld->blockCount,sizeof(real));

    cudaMalloc(&(system->msld->lambdaSite_d),system->msld->blockCount*sizeof(int));
    cudaMalloc(&(system->msld->lambda_d),system->msld->blockCount*sizeof(real));
    cudaMalloc(&(system->msld->lambdaForce_d),system->msld->blockCount*sizeof(real));
    cudaMalloc(&(system->msld->lambdaBias_d),system->msld->blockCount*sizeof(real));
    cudaMalloc(&(system->msld->theta_d),system->msld->blockCount*sizeof(real));
    cudaMalloc(&(system->msld->thetaVelocity_d),system->msld->blockCount*sizeof(real));
    cudaMalloc(&(system->msld->thetaForce_d),system->msld->blockCount*sizeof(real));
    cudaMalloc(&(system->msld->thetaMass_d),system->msld->blockCount*sizeof(real));
    cudaMalloc(&(system->msld->thetaInvsqrtMass_d),system->msld->blockCount*sizeof(real));
    cudaMalloc(&(system->msld->thetaRandom_d),2*system->msld->blockCount*sizeof(real));

    // NYI - this would be a lot easier to read if these were split in to parsing functions.
    fprintf(stdout,"NYI - Initialize all blocks in first site %s:%d\n",__FILE__,__LINE__);
  } else if (strcmp(token,"call")==0) {
    i=io_nexti(line);
    if (i<0 || i>=system->msld->blockCount) {
      fatal(__FILE__,__LINE__,"Error, tried to edit block %d of %d which does not exist.\n",i,system->msld->blockCount-1);
    }
    std::string name=io_nexts(line);
    if (system->selections->selectionMap.count(name)==0) {
      fatal(__FILE__,__LINE__,"Selection %s not found\n",name.c_str());
    }
    for (j=0; j<system->structure->atomList.size(); j++) {
      if (system->selections->selectionMap[name].boolSelection[j]==1) {
        system->msld->atomBlock[j]=i;
      }
    }
// LDIN 3   0.4   0.0   20.0   5.0
// CHARMM: LDIN BLOCK L0 LVEL LMASS LBIAS
// BLOCK SITE THETA THETAV THETAM LBIAS
  } else if (strcmp(token,"initialize")==0) {
#warning "No input guards on data..."
    i=io_nexti(line);
    if (i<0 || i>=system->msld->blockCount) {
      fatal(__FILE__,__LINE__,"Error, tried to edit block %d of %d which does not exist.\n",i,system->msld->blockCount-1);
    }
    system->msld->lambdaSite[i]=io_nexti(line);
    system->msld->theta[i]=io_nextf(line);
    system->msld->thetaVelocity[i]=io_nextf(line);
    system->msld->thetaMass[i]=io_nextf(line);
    system->msld->lambdaBias[i]=io_nextf(line);
    system->msld->lambdaCharge[i]=io_nextf(line);
  } else if (strcmp(token,"gamma")==0) {
    system->msld->gamma=io_nextf(line); // units: ps^-1
  } else if (strcmp(token,"bias")==0) {
    // NYI - add option to reset variable biases
    struct VariableBias vb;
    vb.i=io_nexti(line);
    vb.j=io_nexti(line);
    vb.type=io_nexti(line);
    vb.l0=io_nextf(line);
    vb.k=io_nextf(line);
    vb.n=io_nexti(line);
    system->msld->variableBias_tmp.emplace_back(vb);
  } else if (strcmp(token,"removescaling")==0) {
    std::string name;
    while ((name=io_nexts(line))!="") {
      if (name=="bond") {
        system->msld->scaleTerms[0]=false;
      } else if (name=="urey") {
        system->msld->scaleTerms[1]=false;
      } else if (name=="angle") {
        system->msld->scaleTerms[2]=false;
      } else if (name=="dihe") {
        system->msld->scaleTerms[3]=false;
      } else if (name=="impr") {
        system->msld->scaleTerms[4]=false;
      } else if (name=="cmap") {
        system->msld->scaleTerms[5]=false;
      } else {
        fatal(__FILE__,__LINE__,"Unrecognized token %s. Valid options are bond, urey, angle, dihe, impr, or cmap.\n",name.c_str());
      }
    }
// NYI - restorescaling option to complement remove scaling
  } else if (strcmp(token,"softbond")==0) {
// NYI - check selection length
    std::string name=io_nexts(line);
    int i;
    int j;
    Int2 i2;
    if (system->selections->selectionMap.count(name)==1) {
      j=0;
      for (i=0; i<system->selections->selectionMap[name].boolCount; i++) {
        if (system->selections->selectionMap[name].boolSelection[i]) {
          if (j<2) {
            i2.i[j]=i;
          }
          j++;
        }
      }
      if (j==2) {
        system->msld->softBonds.emplace_back(i2);
      } else {
        fatal(__FILE__,__LINE__,"Found soft bond selection with %d atoms when expected 2 atoms\n",j);
      }
    } else {
      fatal(__FILE__,__LINE__,"Unrecognized token %s used for softbond selection name. Use selection print to see available tokens.\n",name.c_str());
    }
    // NYI - load in soft bond parameters later
  } else if (strcmp(token,"atomrestraint")==0) {
    std::string name=io_nexts(line);
    int i;
    std::vector<int> ar; // NYI - error checking, requires site assignment - ,br,sr; // atom, block, and site
    ar.clear();
    if (system->selections->selectionMap.count(name)==1) {
      for (i=0; i<system->selections->selectionMap[name].boolCount; i++) {
        if (system->selections->selectionMap[name].boolSelection[i]) {
          ar.emplace_back(i);
        }
      }
      system->msld->atomRestraints.emplace_back(ar);
    } else {
      fatal(__FILE__,__LINE__,"Unrecognized token %s used for atomrestraint selection name. Use selection print to see available tokens.\n",name.c_str());
    }
// NYI - charge restraints, put Q in initialize
  } else if (strcmp(token,"print")==0) {
    system->selections->dump();
  } else {
    fatal(__FILE__,__LINE__,"Unrecognized selection token: %s\n",token);
  }
}

int merge_site_block(int site,int block)
{
  if (site>=(1<<16) || block>=(1<<16)) {
    fatal(__FILE__,__LINE__,"Site or block cap of 2^16 exceeded. Site=%d,Block=%d\n",site,block);
  }
  return ((site<<16)|block);
}

// NYI: soft bonds or constrained atom scaling
void Msld::bonded_scaling(int *idx,int *siteBlock,int type,int Nat,int Nsc)
{
  int i,j;
  int ab;
  int block[Nsc+2]={0}; // First term is blockCount, last term is for error checking
  block[0]=blockCount;

  // Sort into a descending list with no duplicates.
  if (scaleTerms[type]) {
    for (i=1; i<Nsc+2; i++) {
      for (j=0; j<Nat; j++) {
        ab=atomBlock[idx[j]];
        if (ab>block[i] && ab<block[i-1]) {
          block[i]=ab;
        }
      }
    }
    // Check for errors
    for (i=1; i<Nsc+1; i++) {
      for (j=i+1; j<Nsc+1; j++) {
        if (block[i]>0 && block[j]>0 && block[i]!=block[j] && lambdaSite[block[i]]==lambdaSite[block[j]]) {
          fatal(__FILE__,__LINE__,"Illegal MSLD scaling between two atoms in the same site (%d) but different blocks (%d and %d)\n",lambdaSite[block[i]],block[i],block[j]);
        }
      }
    }
    if (block[Nsc+1] != 0) {
      fatal(__FILE__,__LINE__,"Only %d lambda scalings allowed in a group of %d bonded atoms\n",Nsc,Nat);
    }
  }

  for (i=0; i<Nsc; i++) {
    siteBlock[i]=merge_site_block(lambdaSite[block[i+1]],block[i+1]);
  }
}

void Msld::bond_scaling(int idx[2],int siteBlock[2])
{
  bonded_scaling(idx,siteBlock,0,2,2);
}

void Msld::ureyb_scaling(int idx[3],int siteBlock[2])
{
  bonded_scaling(idx,siteBlock,1,3,2);
}

void Msld::angle_scaling(int idx[3],int siteBlock[2])
{
  bonded_scaling(idx,siteBlock,2,3,2);
}

void Msld::dihe_scaling(int idx[4],int siteBlock[2])
{
  bonded_scaling(idx,siteBlock,3,4,2);
}

void Msld::impr_scaling(int idx[4],int siteBlock[2])
{
  bonded_scaling(idx,siteBlock,4,4,2);
}

void Msld::cmap_scaling(int idx[8],int siteBlock[3])
{
  bonded_scaling(idx,siteBlock,5,8,3);
}

// Initialize MSLD for a simulation
void Msld::initialize()
{
  int i;

  for (i=1; i<blockCount; i++) {
    thetaInvsqrtMass[i]=1/sqrt(thetaMass[i]);
  }

  // Send the biases over
  send_real(lambdaBias_d,lambdaBias);
  variableBiasCount=variableBias_tmp.size();
  variableBias=(struct VariableBias*)calloc(variableBiasCount,sizeof(struct VariableBias));
  cudaMalloc(&variableBias_d,variableBiasCount*sizeof(struct VariableBias));
  for (i=0; i<variableBiasCount; i++) {
    variableBias[i]=variableBias_tmp[i];
  }
  cudaMemcpy(variableBias_d,variableBias,variableBiasCount*sizeof(struct VariableBias),cudaMemcpyHostToDevice);

  // Get blocksPerSite
  siteCount=1;
  for (i=0; i<blockCount; i++) {
    siteCount=((siteCount>lambdaSite[i])?siteCount:(lambdaSite[i]+1));
    if (i!=0 && lambdaSite[i]!=lambdaSite[i-1] && lambdaSite[i]!=lambdaSite[i-1]+1) {
      fatal(__FILE__,__LINE__,"Blocks must be ordered by consecutive sites. Block %d (site %d) is out of order with block %d (site %d)\n",i,lambdaSite[i],i-1,lambdaSite[i-1]);
    }
  }
  blocksPerSite=(int*)calloc(siteCount,sizeof(int));
  siteBound=(int*)calloc(siteCount+1,sizeof(int));
  cudaMalloc(&blocksPerSite_d,siteCount*sizeof(int));
  cudaMalloc(&siteBound_d,(siteCount+1)*sizeof(int));
  for (i=0; i<blockCount; i++) {
    blocksPerSite[lambdaSite[i]]++;
  }
  if (blocksPerSite[0]!=1) fatal(__FILE__,__LINE__,"Only one block allowed in site 0\n");
  siteBound[0]=0;
  for (i=0; i<siteCount; i++) {
    if (i && blocksPerSite[i]<2) fatal(__FILE__,__LINE__,"At least two blocks are required in each site. %d found at site %d\n",blocksPerSite[i],i);
    siteBound[i+1]=siteBound[i]+blocksPerSite[i];
  }
  cudaMemcpy(blocksPerSite_d,blocksPerSite,siteCount*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(siteBound_d,siteBound,(siteCount+1)*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(lambdaSite_d,lambdaSite,blockCount*sizeof(int),cudaMemcpyHostToDevice);

  // Get lambda on remote node
  send_real(theta_d,theta);
  calc_lambda_from_theta(0); // NYI - pick a stream...
  
//  send_real(system->msld->lambda_d,system->msld->lambda);
}

void Msld::send_real(real *p_d,real *p)
{
  cudaMemcpy(p_d,p,blockCount*sizeof(real),cudaMemcpyHostToDevice);
}

void Msld::recv_real(real *p,real *p_d)
{
  cudaMemcpy(p,p_d,blockCount*sizeof(real),cudaMemcpyDeviceToHost);
}

__global__ void calc_lambda_from_theta_kernel(real *lambda,real *theta,int siteCount,int *siteBound,real fnex)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int j,ji,jf;
  real lLambda;
  real norm=0;

  if (i<siteCount) {
    ji=siteBound[i];
    jf=siteBound[i+1];
    for (j=ji; j<jf; j++) {
#warning "Hardcoded in expf and sinf"
      lLambda=expf(fnex*sinf(theta[j]*ANGSTROM));
      lambda[j]=lLambda;
      norm+=lLambda;
    }
    norm=1/norm;
    for (j=ji; j<jf; j++) {
      lambda[j]*=norm;
    }
  }
}

void Msld::calc_lambda_from_theta(cudaStream_t s)
{
  calc_lambda_from_theta_kernel<<<(siteCount+BLMS-1)/BLMS,BLMS,0,s>>>(lambda_d,theta_d,siteCount,siteBound_d,fnex);
}

__global__ void calc_thetaForce_from_lambdaForce_kernel(real *lambda,real *theta,real *lambdaForce,real *thetaForce,int blockCount,int *lambdaSite,int *siteBound,real fnex)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int j, ji, jf;
  real li, fi;

  if (i<blockCount) {
    li=lambda[i];
    fi=lambdaForce[i];
    ji=siteBound[lambdaSite[i]];
    jf=siteBound[lambdaSite[i]+1];
    for (j=ji; j<jf; j++) {
      fi+=-lambda[j]*lambdaForce[j];
    }
    fi*=li*fnex*cosf(ANGSTROM*theta[i])*ANGSTROM;
    thetaForce[i]=fi;
  }
}

void Msld::calc_thetaForce_from_lambdaForce(cudaStream_t s)
{
  calc_thetaForce_from_lambdaForce_kernel<<<(blockCount+BLMS-1)/BLMS,BLMS,0,s>>>(lambda_d,theta_d,lambdaForce_d,thetaForce_d,blockCount,lambdaSite_d,siteBound_d,fnex);
}

__global__ void calc_fixedBias_kernel(real *lambda,real *lambdaBias,real *lambdaForce,real *energy,int blockCount)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  extern __shared__ real sEnergy[];
  real lEnergy=0;

  if (i<blockCount) {
    realAtomicAdd(&lambdaForce[i],lambdaBias[i]);
    if (energy) {
      lEnergy=lambdaBias[i]*lambda[i];
    }
  }

  // Energy, if requested
  if (energy) {
    __syncthreads();
    lEnergy+=__shfl_down_sync(0xFFFFFFFF,lEnergy,1);
    lEnergy+=__shfl_down_sync(0xFFFFFFFF,lEnergy,2);
    lEnergy+=__shfl_down_sync(0xFFFFFFFF,lEnergy,4);
    lEnergy+=__shfl_down_sync(0xFFFFFFFF,lEnergy,8);
    lEnergy+=__shfl_down_sync(0xFFFFFFFF,lEnergy,16);
    __syncthreads();
    if ((0x1F & threadIdx.x)==0) {
      sEnergy[threadIdx.x>>5]=lEnergy;
    }
    __syncthreads();
    if (threadIdx.x < (blockDim.x>>5)) {
      lEnergy=sEnergy[threadIdx.x];
      lEnergy+=__shfl_down_sync(0xFFFFFFFF,lEnergy,1);
      lEnergy+=__shfl_down_sync(0xFFFFFFFF,lEnergy,2);
      lEnergy+=__shfl_down_sync(0xFFFFFFFF,lEnergy,4);
    }
    if (threadIdx.x==0) {
      realAtomicAdd(energy,lEnergy);
    }
  }
}

void Msld::calc_fixedBias(System *system,bool calcEnergy)
{
  cudaStream_t s=0;
  real *pEnergy=NULL;
  int shMem=0;

  if (calcEnergy) {
    shMem=BLMS*sizeof(real)/32;
    pEnergy=system->state->energy_d+eelambda;
  }
  if (system->potential) {
    s=system->potential->biaspotStream;
  }

  calc_fixedBias_kernel<<<(blockCount+BLMS-1)/BLMS,BLMS,shMem,s>>>(lambda_d,lambdaBias_d,lambdaForce_d,pEnergy,blockCount);
}

__global__ void calc_variableBias_kernel(real *lambda,real *lambdaForce,real *energy,int variableBiasCount,struct VariableBias *variableBias)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  struct VariableBias vb;
  real li,lj;
  real fi,fj;
  extern __shared__ real sEnergy[];
  real lEnergy=0;

  if (i<variableBiasCount) {
    vb=variableBias[i];
    li=lambda[vb.i];
    lj=lambda[vb.j];
    if (vb.type==6) {
      lEnergy=vb.k*li*lj;
      fi=vb.k*lj;
      fj=vb.k*li;
    } else if (vb.type==8) {
      lEnergy=vb.k*li*lj/(li+vb.l0);
      fi=vb.k*vb.l0*lj/((li+vb.l0)*(li+vb.l0));
      fj=vb.k*li/(li+vb.l0);
    } else if (vb.type==10) {
      lEnergy=vb.k*lj*(1-expf(vb.l0*li));
      fi=vb.k*lj*(-vb.l0*expf(vb.l0*li));
      fj=vb.k*(1-expf(vb.l0*li));
    } else {
#warning "Need error checking to make sure this doesn't happen"
      lEnergy=NAN;
      fi=NAN;
      fj=NAN;
    }
    realAtomicAdd(&lambdaForce[vb.i],fi);
    realAtomicAdd(&lambdaForce[vb.j],fj);
  }

  // Energy, if requested
  if (energy) {
    __syncthreads();
    lEnergy+=__shfl_down_sync(0xFFFFFFFF,lEnergy,1);
    lEnergy+=__shfl_down_sync(0xFFFFFFFF,lEnergy,2);
    lEnergy+=__shfl_down_sync(0xFFFFFFFF,lEnergy,4);
    lEnergy+=__shfl_down_sync(0xFFFFFFFF,lEnergy,8);
    lEnergy+=__shfl_down_sync(0xFFFFFFFF,lEnergy,16);
    __syncthreads();
    if ((0x1F & threadIdx.x)==0) {
      sEnergy[threadIdx.x>>5]=lEnergy;
    }
    __syncthreads();
    if (threadIdx.x < (blockDim.x>>5)) {
      lEnergy=sEnergy[threadIdx.x];
      lEnergy+=__shfl_down_sync(0xFFFFFFFF,lEnergy,1);
      lEnergy+=__shfl_down_sync(0xFFFFFFFF,lEnergy,2);
      lEnergy+=__shfl_down_sync(0xFFFFFFFF,lEnergy,4);
    }
    if (threadIdx.x==0) {
      realAtomicAdd(energy,lEnergy);
    }
  }
}

void Msld::calc_variableBias(System *system,bool calcEnergy)
{
  cudaStream_t s=0;
  real *pEnergy=NULL;
  int shMem=0;

  if (calcEnergy) {
    shMem=BLMS*sizeof(real)/32;
    pEnergy=system->state->energy_d+eelambda;
  }
  if (system->potential) {
    s=system->potential->biaspotStream;
  }

  calc_variableBias_kernel<<<(variableBiasCount+BLMS-1)/BLMS,BLMS,shMem,s>>>(lambda_d,lambdaForce_d,pEnergy,variableBiasCount,variableBias_d);
}
