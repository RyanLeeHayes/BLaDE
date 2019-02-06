#ifndef MD_RNA_IONS_NBLIST_H
#define MD_RNA_IONS_NBLIST_H

#include <stdio.h>

#include "defines.h"

#include "vec.h"
#include "md.h"

typedef struct struct_bin
{
  int i;
  int bx;
  int by;
  real z;
//   vec c;
} struct_bin;


typedef struct struct_ij
{
  int i;
  int j;
} struct_ij;


typedef struct struct_nblist
{
  struct_bin *bin;
  // struct_bin div;
  int N; // Number of particles
  int Nbit;
  int Nblocks;
  int *iblock; // length N (only first Nblocks matter) - beginning index within block
  // int *Nblock; // length N (only first Nblocks matter) - number of entries after iblock in block // get from iblock[i+1]-iblock[i]
  // int *block; // length N
  int TN; // Tile size (TLJ or TES)
  int Ntiles;
  int mNtiles; // Ntiles allocated. Call realloc if Ntiles>mNtiles
  // int *itile; // length mNtiles. bin[iblock[itile[i]]].i is the first i atom index in the ith tile
  // int *jtile; // length mNtiles. bin[iblock[jtile[i]]].i is the first j atom index in the ith tile
  struct_ij *ijtile; // combination of itile and jtile
  // int *iB; // mNtiles - the block indices: i is bin[iblock[blocal[iB[blockIdx]]]+threadIdx].i
  // int *jB; // mNtiles - the block indices: j is bin[iblock[blocal[iB[blockIdx]]]+threadIdx].i
  // int *i; // mNtiles x TNB - the atom indices i
  // int *j; // mNtiles x TNB - the atom indices j
  // int *local; // mNtiles x 2*TNB - maps the local atom indices i and j back to global coordinates
  // int *blocal; // mNtiles x 2 - the block a tile corresponds to. bin[iblock[blocal[local[i]]]].i is the actual atom index
  // int *N_local; // (mNtiles+BNB/TNB-1)/(BNB/TNB) - number of tiles worth of unique coordinates in each block
  vec *shift;
  unsigned int *nlist; // mNtiles x TNB x TNB array of boolean values packed into larger chuncks for fast loading. TNB is the NonBonded Block size (see defines.h)
  // int *debug;
} struct_nblist;

void alloc_nblist(int N,int TN,struct_nblist** H4H,struct_nblist** H4D);
void free_nblist(struct_nblist* H4H,struct_nblist* H4D);
void neighborsearch(struct_md* md);
void neighborsearch_elec(struct_md* md);

void dump_nblist(FILE *fp,struct_nblist *H4H,struct_nblist *H4D,real *xhost,real *xdevice,int N,int step);

#endif
