#ifndef MD_RNA_IONS_ATOMS_H
#define MD_RNA_IONS_ATOMS_H

#include "defines.h"
#include "leapparms.h"
#include "mersenne.h"

struct struct_collate;

typedef struct struct_atoms
{
  int N; // Number of coordinates, 3 times number of atoms
  int Ni; // atoms before this are frozen
  int Nf; // This and later atoms are frozen
  real *x;
  // real *shift;
  real *v;
  real *vhalf;
  real *f;
  real *m;
  real *misqrt;
  real *Vs_delay;
  unsignedfixreal *fixx;
  real *fixreal2real;
  real *real2fixreal;
  struct_leapparms lp;
  struct_mtstate mts[2];
} struct_atoms;

struct_atoms* alloc_atoms(int N,int Ni,int Nf,struct_leapparms lp,cudaStream_t strm,int pbcflag);

void free_atoms(struct_atoms* atoms);

__global__ void setfloat2fix(struct_atoms at,real* box);
__global__ void float2fix(struct_atoms at);
__global__ void fix2float(struct_atoms at);

#endif

