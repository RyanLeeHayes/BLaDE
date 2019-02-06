#ifndef MD_RNA_ION_SORT_H
#define MD_RNA_ION_SORT_H

#include "defines.h"

#include "topol.h"
#include "nblist.h"
/*
// Number of threads in each block
#define SORT_THREADS_PER_BLOCK 32
// Number of bonds each thread is responsible for
#define SORT_ATOMS_PER_THREAD 32

typedef struct struct_sort {
  int N;
  int n_keys;
  int n_data;
  int *keys;
  int *keys_buf;
  void *data;
  void *data_buf;
  int (*hist)[SORT_THREADS_PER_BLOCK];
} struct_sort;
*/

typedef struct struct_bondblock {
  int *local;
  int *N_local;
} struct_bondblock;

typedef struct struct_angleblock {
  int *local;
  int *N_local;
} struct_angleblock;

typedef struct struct_dihblock {
  int *local;
  int *N_local;
} struct_dihblock;

struct_bondblock* alloc_bondblock(struct_bondparms* bondparms,int N);
void free_bondblock(struct_bondblock* bondblock);

struct_angleblock* alloc_angleblock(struct_angleparms* angleparms,int N);
void free_angleblock(struct_angleblock* angleblock);

struct_dihblock* alloc_dihblock(struct_dihparms* dihparms,int N);
void free_dihblock(struct_dihblock* dihblock);

#endif
