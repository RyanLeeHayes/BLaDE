#ifndef MD_RNA_IONS_PARMS_H
#define MD_RNA_IONS_PARMS_H

#include <stdio.h>

#include "defines.h"


#include "times.h"

struct struct_leapparms;
struct struct_bondparms;
struct struct_bondblock;
struct struct_angleparms;
struct struct_angleblock;
struct struct_dihparms;
struct struct_dihblock;
struct struct_pair1parms;
struct struct_pair5parms;
struct struct_pair6parms;
struct struct_pair7parms;
struct struct_pair8parms;
struct struct_nbparms;
struct struct_exclparms;
struct struct_exclhash;

typedef struct struct_parms
{
  int id;
  int offset;
  int phase;
  real maxh;
  gmx_cycles_t maxc;
  real kB;
  real kT;
  real arg_box;
  char *arg_topfile;
  char *arg_paramfile;
  char *arg_grofile;
  char *arg_outdir;
  char *arg_biasfile;
  int trajfmt; // 0 is gro file, 1 is xtc
  real outputguard; // approximate output limit
  real k12;
  real s2[2][2];
  real rcother;
  int t_output;
  int t_ns;
  int steplim;
  int stepmeta;
  int umbrella_function;
  int hessian;
  real bias_A;
  real bias_sigQ;
  real bias_sigFQ;
  real dt;
  struct struct_leapparms *leapparms;
  struct struct_leapparms *Qleapparms;
  int kbgo; // flag - 0 gromacs SMOG / 1 charmm kbgo
  real kbgo_dih0;
  int N_bond;
  int N_angle;
  int N_dih;
  int N_pair1;
  int N_pair5;
  int N_pair6;
  int N_pair7;
  int N_pair8;
  int N_nb;
  int N_excl;
  struct struct_bondparms *bondparms;
  struct struct_bondblock *bondblock;
  struct struct_angleparms *angleparms;
  struct struct_angleblock *angleblock;
  struct struct_dihparms *dihparms;
  struct struct_dihblock *dihblock;
  struct struct_pair1parms *pair1parms;
  struct struct_pair5parms *pair5parms;
  struct struct_pair6parms *pair6parms;
  struct struct_pair7parms *pair7parms;
  struct struct_pair8parms *pair8parms;
  struct struct_umbrella *umbrella;
  struct struct_nbparms *nbparms;
  struct struct_exclparms *exclparms;
  // struct struct_exclparms *exclparmsdev;
  struct struct_exclhash *exclhash;
  struct struct_exclhash *exclhashdev;
} struct_parms;

struct_parms* alloc_parms(int argc, char *argv[], FILE *log);

void free_parms(struct_parms* parms);

#endif

