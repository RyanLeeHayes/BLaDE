#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>

#include "domdec/domdec.h"
#include "system/system.h"
#include "system/state.h"
#include "system/potential.h"
#include "run/run.h"
#include "main/real3.h"
#include "main/gpu_check.h"



__global__ void assign_blocks_count_cells_kernel(
  int globalCount,int3 gridDomdec,int2 domainDiv,int cellDivZ,int *domain,
  real3 *position,real3 box,int *atomCell,int *cellAtomCount)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;

  if (i<globalCount) {
    int domainIdx=domain[i];
    int3 idDomdec;
    idDomdec.x=domainIdx/(gridDomdec.y*gridDomdec.z);
    idDomdec.y=domainIdx/gridDomdec.z-idDomdec.x*gridDomdec.y;
    idDomdec.z=domainIdx-idDomdec.x*gridDomdec.y*gridDomdec.z-idDomdec.y*gridDomdec.z;

    real3 xi=position[i];
    int ix=(int)floor((xi.x*gridDomdec.x/box.x-idDomdec.x)*domainDiv.x);
    int iy=(int)floor((xi.y*gridDomdec.y/box.y-idDomdec.y)*domainDiv.y);
    int iz=(int)floor((xi.z*gridDomdec.z/box.z-idDomdec.z)*cellDivZ);
    ix=(ix>=domainDiv.x?domainDiv.x-1:ix);
    iy=(iy>=domainDiv.y?domainDiv.y-1:iy);
    iz=(iz>=cellDivZ?cellDivZ-1:iz);
    ix=(ix<0?0:ix);
    iy=(iy<0?0:iy);
    iz=(iz<0?0:iz);

    int column=(domainIdx*domainDiv.x+ix)*domainDiv.y+iy;
    int cell=column*cellDivZ+iz;
    atomCell[i]=cell;
    atomicAdd(&cellAtomCount[cell],1);
  }
}

__global__ void assign_blocks_count_column_blocks_kernel(
  int columnCount,int cellDivZ,int *cellAtomOffset,int *blocksPerColumn)
{
  int column=blockIdx.x*blockDim.x+threadIdx.x;

  if (column<columnCount) {
    int firstCell=column*cellDivZ;
    int atomBegin=(firstCell==0?0:cellAtomOffset[firstCell-1]);
    int atomEnd=cellAtomOffset[firstCell+cellDivZ-1];
    blocksPerColumn[column]=(atomEnd-atomBegin+31)/32;
  }
}

__global__ void assign_blocks_scan_bounds_kernel(
  int columnCount,int columnsPerDomain,int cellDivZ,int globalCount,
  int *cellAtomOffset,int *cumulativeBlocks,int *blockCount,int *blockBounds)
{
  int column=blockIdx.x*blockDim.x+threadIdx.x;

  if (column<columnCount) {
    int firstCell=column*cellDivZ;
    int atomBegin=(firstCell==0?0:cellAtomOffset[firstCell-1]);
    int blockBegin=(column==0?0:cumulativeBlocks[column-1]);
    int blockEnd=cumulativeBlocks[column];

    for (int block=blockBegin; block<blockEnd; block++) {
      blockBounds[block]=atomBegin+32*(block-blockBegin);
    }

    if (column==0) {
      blockCount[0]=0;
    }
    if ((column+1)%columnsPerDomain==0) {
      blockCount[(column+1)/columnsPerDomain]=blockEnd;
    }
    if (column==columnCount-1) {
      blockBounds[blockEnd]=globalCount;
    }
  }
}

__global__ void assign_blocks_prepare_cell_cursors_kernel(
  int cellCount,int *cellAtomOffset,int *cellAtomCursor)
{
  int cell=blockIdx.x*blockDim.x+threadIdx.x;

  if (cell<cellCount) {
    cellAtomCursor[cell]=(cell==0?0:cellAtomOffset[cell-1]);
  }
}

__global__ void assign_blocks_scatter_kernel(
  int globalCount,int cellDivZ,int *atomCell,int *cellAtomCursor,
  int *cellAtomOffset,int *cumulativeBlocks,int *localToGlobal,
  int *globalToLane,int *globalToBlock,NbondPotential *nbonds,
  NbondPotential *localNbonds)
{
  int iGlobal=blockIdx.x*blockDim.x+threadIdx.x;

  if (iGlobal<globalCount) {
    int cell=atomCell[iGlobal];
    int column=cell/cellDivZ;
    int firstCell=column*cellDivZ;
    int atomBegin=(firstCell==0?0:cellAtomOffset[firstCell-1]);
    int blockBegin=(column==0?0:cumulativeBlocks[column-1]);
    int iLocal=atomicAdd(&cellAtomCursor[cell],1);
    int rank=iLocal-atomBegin;
    int lane=rank&31;
    int block=blockBegin+rank/32;

    localToGlobal[iLocal]=iGlobal;
    globalToLane[iGlobal]=lane;
    globalToBlock[iGlobal]=block;
    localNbonds[32*block+lane]=nbonds[iGlobal];
  }
}

/* OLD __global__ void assign_blocks_finish_local_kernel(int globalCount,int *localToGlobal,real3 *position,real3 *localPosition)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<globalCount) {
    int iGlobal=localToGlobal[i];
    localPosition[i]=position[iGlobal];
  }
}*/

//    assign_blocks_finish_local_kernel<<<(32*blockCount+BLUP-1)/BLUP,BLUP,0,system->update->updateStream>>>(blockCount,blockBounds_d,localToGlobal_d,(real3*)system->state->position_d,localPosition_d,blockVolume_d);
__global__ void assign_blocks_localPosition_kernel(int blockCount,int *blockBounds,int *localToGlobal,real3 *position,real3 *localPosition,struct DomdecBlockVolume *blockVolume)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int iBlock=i/32;
  int j,iLocal,iGlobal,atomsInBlock;
  real3 xi;
  struct DomdecBlockVolume v,vCompare;

  if (iBlock<blockCount) {
    // Copy over positions to local structure
    iLocal=blockBounds[iBlock];
    atomsInBlock=blockBounds[iBlock+1]-iLocal;
    iLocal+=(i&31);
    if ((i&31)<atomsInBlock) {
      iGlobal=localToGlobal[iLocal];
      xi=position[iGlobal];
      localPosition[i]=xi;
    }

    // Find extreme values
    v.max=xi;
    v.min=xi;
    for (j=1; j<32; j*=2) {
      vCompare.min.x=__shfl_down_sync(0xFFFFFFFF,v.min.x,j);
      vCompare.min.y=__shfl_down_sync(0xFFFFFFFF,v.min.y,j);
      vCompare.min.z=__shfl_down_sync(0xFFFFFFFF,v.min.z,j);
      vCompare.max.x=__shfl_down_sync(0xFFFFFFFF,v.max.x,j);
      vCompare.max.y=__shfl_down_sync(0xFFFFFFFF,v.max.y,j);
      vCompare.max.z=__shfl_down_sync(0xFFFFFFFF,v.max.z,j);
      if ((i&31)+j<atomsInBlock) {
        v.min.x=(v.min.x<vCompare.min.x?v.min.x:vCompare.min.x);
        v.min.y=(v.min.y<vCompare.min.y?v.min.y:vCompare.min.y);
        v.min.z=(v.min.z<vCompare.min.z?v.min.z:vCompare.min.z);
        v.max.x=(v.max.x>vCompare.max.x?v.max.x:vCompare.max.x);
        v.max.y=(v.max.y>vCompare.max.y?v.max.y:vCompare.max.y);
        v.max.z=(v.max.z>vCompare.max.z?v.max.z:vCompare.max.z);
      }
    }
    if ((i&31)==0) {
      blockVolume[iBlock]=v;
    }
  }
}

void Domdec::assign_blocks(System *system)
{
  Run *r=system->run;

  if (id>=0) { 
    real3 box;
    if (system->state->typeBox) {
      box.x=system->state->tricBox_f.a.x;
      box.y=system->state->tricBox_f.b.y;
      box.z=system->state->tricBox_f.c.z;
    } else {
      box=system->state->orthBox_f;
    }

    int columnsPerDomain=domainDiv.x*domainDiv.y;
    int columnCount=idCount*columnsPerDomain;

    gpuCheck(cudaMemsetAsync(cellAtomCount_d,0,blockCellCount*sizeof(int),r->updateStream));
    assign_blocks_count_cells_kernel<<<(globalCount+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(
      globalCount,gridDomdec,domainDiv,cellDivZ,domain_d,
      (real3*)system->state->position_fd,box,atomCell_d,cellAtomCount_d);
    gpuCheck(cudaGetLastError());

    gpuCheck(cub::DeviceScan::InclusiveSum(scanTemp_d,scanTempBytes,
      cellAtomCount_d,cellAtomOffset_d,blockCellCount,r->updateStream));

    assign_blocks_count_column_blocks_kernel<<<(columnCount+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(
      columnCount,cellDivZ,cellAtomOffset_d,blocksPerColumn_d);
    gpuCheck(cudaGetLastError());

    gpuCheck(cub::DeviceScan::InclusiveSum(scanTemp_d,scanTempBytes,
      blocksPerColumn_d,blocksPerColumn_d,columnCount,r->updateStream));

    assign_blocks_scan_bounds_kernel<<<(columnCount+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(
      columnCount,columnsPerDomain,cellDivZ,globalCount,cellAtomOffset_d,
      blocksPerColumn_d,blockCount_d,blockBounds_d);
    gpuCheck(cudaGetLastError());

    assign_blocks_prepare_cell_cursors_kernel<<<(blockCellCount+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(
      blockCellCount,cellAtomOffset_d,cellAtomCount_d);
    gpuCheck(cudaGetLastError());

    assign_blocks_scatter_kernel<<<(globalCount+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(
      globalCount,cellDivZ,atomCell_d,cellAtomCount_d,cellAtomOffset_d,
      blocksPerColumn_d,localToGlobal_d,globalToLane_d,globalToBlock_d,
      system->potential->nbonds_d,localNbonds_d);
    gpuCheck(cudaGetLastError());

    gpuCheck(cudaMemcpy(blockCount,blockCount_d,(idCount+1)*sizeof(int),cudaMemcpyDeviceToHost));

    // Redundant with pack_positions, needed for call to cull
    assign_blocks_localPosition_kernel<<<(32*blockCount[idCount]+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(blockCount[idCount],blockBounds_d,localToGlobal_d,(real3*)system->state->position_fd,localPosition_d,blockVolume_d);
    gpuCheck(cudaGetLastError());
  }
}

void Domdec::pack_positions(System *system)
{
  Run *r=system->run;
  int N=blockCount[idCount];
  if (id>=0) {
    assign_blocks_localPosition_kernel<<<(32*N+BLUP-1)/BLUP,BLUP,0,r->nbdirectStream>>>(N,blockBounds_d,localToGlobal_d,(real3*)system->state->position_fd,localPosition_d,blockVolume_d);
    gpuCheck(cudaGetLastError());
  }
}

__global__ void unpack_forces_kernel(int blockCount,int *blockBounds,int *localToGlobal,real3_f *force,real3_f *localForce)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int iBlock=i/32;
  int iLocal,atomsInBlock;

  if (iBlock<blockCount) {
    iLocal=blockBounds[iBlock];
    atomsInBlock=blockBounds[iBlock+1]-iLocal;
    iLocal+=(i&31);
    if ((i&31)<atomsInBlock) {
      at_real3_inc(&force[localToGlobal[iLocal]],localForce[i]);
    }
  }
}

void Domdec::unpack_forces(System *system)
{
  Run *r=system->run;
  int N=blockCount[idCount];
  if (id>=0) {
    unpack_forces_kernel<<<(32*N+BLUP-1)/BLUP,BLUP,0,r->nbdirectStream>>>(N,blockBounds_d,localToGlobal_d,(real3_f*)system->state->force_d,localForce_d);
    gpuCheck(cudaGetLastError());
  }
}
