#ifndef MD_RNA_IONS_TOPOL_H
#define MD_RNA_IONS_TOPOL_H

#include "defines.h"

#include "md.h"
#include "parms.h"

typedef struct struct_bondparms
{
  int i;
  int j;
  real k0;
  real r0;
} struct_bondparms;

typedef struct struct_angleparms
{
  int i;
  int j;
  int k;
  real k0;
  real t0;
} struct_angleparms;

typedef struct struct_dihparms
{
  int i;
  int j;
  int k;
  int l;
  real k0;
  real p0;
  real n;
  real k02;
  real p02;
  real n2;
} struct_dihparms;

typedef struct struct_pair1parms
{
  int i;
  int j;
  real C6;
  real C12;
} struct_pair1parms;

typedef struct struct_pair5parms
{
  int i;
  int j;
  real eps;
  real r0;
  real sigma;
} struct_pair5parms;

typedef struct struct_pair6parms
{
  int i;
  int j;
  real eps;
  real r0;
  real sigma;
  real eps12;
} struct_pair6parms;

typedef struct struct_pair7parms
{
  int i;
  int j;
  real eps;
  real r01;
  real sigma1;
  real r02;
  real sigma2;
  real eps12;
} struct_pair7parms;

typedef struct struct_pair8parms
{
  int i;
  int j;
  real amp;
  real mu;
  real sigma;
} struct_pair8parms;

typedef struct struct_umbrella
{
int freq_out;
real k_umb;
real Q_ref;
real Q_start;
real Q_steps;
real *Q;
real *EQ;
int type;
} struct_umbrella ;

typedef struct struct_nbparms
{
  real eps;
  real rmin;
} struct_nbparms;
  
typedef struct struct_exclparms
{ // i is assumed by index of pointer.
  int *j;
  int N[3];
  int Nmax;
} struct_exclparms;

typedef struct struct_exclhash
{
  int Nind;
  int *ind;
  int Ndata;
  int *data;
} struct_exclhash;

typedef enum {
  eT_none,
  eT_bond,
  eT_angle,
  eT_dih,
  eT_pair1,
  eT_pair5,
  eT_pair6,
  eT_pair7,
  eT_pair8,
  eT_nb
} eTOP;


// struct_parms* alloc_topparms_smog(struct_md *md);
// void free_topparms_smog(struct_parms *parms);

// struct_parms* alloc_topparms_kbgo(struct_md *md);
// void free_topparms_kbgo(struct_parms *parms);

struct_parms* alloc_topparms(struct_md *md);
void free_topparms(struct_parms *parms);

// struct_exclhash* make_empty_exclhash(int N);
// struct_exclhash* alloc_exclhashdev(struct_exclhash* exclhash);

#endif

