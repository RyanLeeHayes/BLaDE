#include <cuda_runtime.h>

#include "system/potential.h"
#include "system/system.h"
#include "io/io.h"
#include "system/parameters.h"
#include "system/structure.h"
#include "msld/msld.h"
#include "system/state.h"
#include "run/run.h"
#include "bonded/bonded.h"



// Class constructors
Potential::Potential() {
  bondCount=0;
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
}

Potential::~Potential()
{
  if (bonds) free(bonds);
  if (angles) free(angles);
  if (dihes) free(dihes);
  if (imprs) free(imprs);
  if (cmaps) free(cmaps);
  if (bonds_d) cudaFree(bonds);
  if (angles_d) cudaFree(angles);
  if (dihes_d) cudaFree(dihes);
  if (imprs_d) cudaFree(imprs);
  if (cmaps_d) cudaFree(cmaps);
}



// Methods
void Potential::initialize(System *system)
{
  int i,j;
  Parameters *param=system->parameters;
  Structure *struc=system->structure;
  Msld *msld=system->msld;

  bonds_tmp.clear();
  angles_tmp.clear();
  dihes_tmp.clear();
  imprs_tmp.clear();
  cmaps_tmp.clear();

  for (i=0; i<struc->bondList.size(); i++) {
    TypeName2 type;
    struct BondPotential bond;
    struct BondParameter bp;
    // Get participating atoms
    for (j=0; j<2; j++) {
      bond.idx[j]=struc->bondList[i].i[j];
      type.t[j]=struc->atomList[bond.idx[j]].atomTypeName;
    }
    // Get their MSLD scaling
    msld->bond_scaling(bond.idx,bond.siteBlock);
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
    } else {
      bonds_tmp.emplace_back(bond);
    }
  }

  for (i=0; i<struc->angleList.size(); i++) {
    TypeName3 type;
    struct BondPotential bond; // urey bradley
    struct AnglePotential angle;
    struct AngleParameter ap;
    // Get participating atoms
    for (j=0; j<3; j++) {
      angle.idx[j]=struc->angleList[i].i[j];
      type.t[j]=struc->atomList[angle.idx[j]].atomTypeName;
    }
    bond.idx[0]=angle.idx[0];
    bond.idx[1]=angle.idx[2];
    // Get their MSLD scaling
    msld->angle_scaling(angle.idx,angle.siteBlock);
    msld->ureyb_scaling(angle.idx,bond.siteBlock);
    // Get their parameters
    if (param->angleParameter.count(type)==0) {
      fatal(__FILE__,__LINE__,"No angle parameter for %6d(%6s) %6d(%6s) %6d(%6s)\n",angle.idx[0],type.t[0].c_str(),angle.idx[1],type.t[1].c_str(),angle.idx[2],type.t[2].c_str());
    }
    ap=param->angleParameter[type];
    angle.kangle=ap.kangle;
    angle.angle0=ap.angle0;
    bond.kb=ap.kureyb;
    bond.b0=ap.ureyb0;
    angles_tmp.emplace_back(angle);
    // Only include urey bradley if it's nonzero
    if (bond.kb != 0) {
      bonds_tmp.emplace_back(bond);
    }
  }

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
    msld->dihe_scaling(dihe.idx,dihe.siteBlock);
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
    // Include each harmonic as a separate dihedral.
    for (j=0; j<dp.size(); j++) {
      dihe.kdih=dp[0].kdih;
      dihe.ndih=dp[0].ndih;
      dihe.dih0=dp[0].dih0;
      dihes_tmp.emplace_back(dihe);
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
    msld->impr_scaling(impr.idx,impr.siteBlock);
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
    impr.kimp=ip.kimp;
    impr.imp0=ip.imp0;
    imprs_tmp.emplace_back(impr);
  }

/*
#warning "Missing CMAP terms"
#warning "Missing nonbonded terms"
  if (atomTypeIdx) free(atomTypeIdx);
  atomTypeIdx=(int*)calloc(atomCount,sizeof(int));
  if (charge) free(charge);
  charge=(real*)calloc(atomCount,sizeof(real));
  if (mass) free(mass);
  mass=(real*)calloc(atomCount,sizeof(real));


  for (i=0; i<atomCount; i++) {
    struct AtomStructure at=atomList[i];
    param->require_type_name(at.atomTypeName,"searching for atom types in structure setup");
    atomTypeIdx[i]=param->atomTypeMap[at.atomTypeName];
    charge[i]=at.charge;
    mass[i]=param->atomMass[at.atomTypeName];
  }
*/
  fprintf(stdout,"Implement nonbonded structures too\n");

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

  // for (i=0; i<5; i++) {
  //   cudaStreamCreate(&bondedStream[i]);
  //   cudaEventCreate(&bondedComplete[i]);
  // }
  cudaStreamCreate(&bondedStream);
  cudaStreamCreate(&nbdirectStream);
  cudaStreamCreate(&nbrecipStream);
// WORKING HERE, add everything to bonds_tmp, remove hbonds if appropriate, put urey bradleys in too if they exit. Also create a structure for constrained bonds.

  // forceComplete=bondedComplete[i];
}

void Potential::calc_force(int step,System *system)
{
  bool calcEnergy=(step%system->run->freqNRG==0);
  // fprintf(stdout,"Force calculation placeholder (step %d)\n",step);


  if (calcEnergy) {
    cudaMemset(system->state->energy_d,0,eeend*sizeof(real));
  }

  getforce_bond(system,calcEnergy);
  getforce_angle(system,calcEnergy);
  getforce_dihe(system,calcEnergy);
  getforce_impr(system,calcEnergy);

  // cudaEventRecord(forceComplete,bondedStream[0]);
  cudaDeviceSynchronize();
}

