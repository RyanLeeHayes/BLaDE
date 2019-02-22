#include <string.h>

#include "msld/msld.h"
#include "system/system.h"
#include "io/io.h"
#include "system/selections.h"
#include "system/structure.h"



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
  lambdaCharge=NULL;

  lambda_d=NULL;
  lambdaForce_d=NULL;
  theta_d=NULL;
  thetaVelocity_d=NULL;
  thetaMass_d=NULL;

  variableBias.clear();
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
  if (lambdaCharge) free(lambdaCharge);

  if (lambda_d) cudaFree(lambda_d);
  if (lambdaForce_d) cudaFree(lambdaForce_d);
  if (theta_d) cudaFree(theta_d);
  if (thetaVelocity_d) cudaFree(thetaVelocity_d);
  if (thetaMass_d) cudaFree(thetaMass_d);
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
    system->msld->lambdaCharge=(real*)calloc(system->msld->blockCount,sizeof(real));

    cudaMalloc(&(system->msld->lambda_d),system->msld->blockCount*sizeof(real));
    cudaMalloc(&(system->msld->lambdaForce_d),system->msld->blockCount*sizeof(real));
    cudaMalloc(&(system->msld->theta_d),system->msld->blockCount*sizeof(real));
    cudaMalloc(&(system->msld->thetaVelocity_d),system->msld->blockCount*sizeof(real));
    cudaMalloc(&(system->msld->thetaMass_d),system->msld->blockCount*sizeof(real));

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
  } else if (strcmp(token,"bias")==0) {
    // NYI - add option to reset variable biases
    struct VariableBias vb;
    vb.i=io_nexti(line);
    vb.j=io_nexti(line);
    vb.type=io_nexti(line);
    vb.l0=io_nextf(line);
    vb.k=io_nextf(line);
    vb.n=io_nexti(line);
    system->msld->variableBias.emplace_back(vb);
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

void Msld::send_real(real *p_d,real *p)
{
  cudaMemcpy(p_d,p,blockCount*sizeof(real),cudaMemcpyHostToDevice);
}

void Msld::recv_real(real *p,real *p_d)
{
  cudaMemcpy(p,p_d,blockCount*sizeof(real),cudaMemcpyDeviceToHost);
}
