#ifndef MD_RNA_IONS_STATE_H
#define MD_RNA_IONS_STATE_H

#include "defines.h"

#include "vec.h"
#include "md.h"

struct struct_qlist;
struct struct_atoms;
// struct struct_collate;
// struct struct_nlist;
struct struct_nblist;
struct struct_mtstate;

typedef enum {
  eE_bond,
  eE_angle,
  eE_dihedral,
  eE_pair1,
  eE_pair5,
  eE_pair6,
  eE_pair7,
  eE_pair8,
  eE_elec,
  eE_lj,
  eE_MAX
} eNRG;

typedef struct struct_bias
{
  // Scalars
  real *Q;
  real *Qhost;
  real *gUgQ;
  real *gUgQhost;
  real *gQgQ;
  real *gQgQhost;
  real *ggQ;
  real *ggQhost;
  real *gQggQgQ;
  real *gQggQgQhost;
  real *FQ;
  real *FQhost;
  real *gQgFQ;
  real *gQgFQhost;
  // Vectors
  real *gQ;
  real *gFQ;
  real *dotwggU;
  real *dotwggQ;
  real *ggQgQ;
  // Biasing
  real *B;
  real *dBdQ;
  real *dBdFQ;
  real *Bhost;
  real *dBdQhost;
  real *dBdFQhost;
  // Adaptive
  real *history;
  real *history_buffer;
  int Nhistory;
  int ihistory;
  // real linearbias;
  // Parameters for adaptive bias
  int type; // umbrella_function
  real A;
  real sigQ;
  real sigFQ;
} struct_bias;

typedef struct struct_state
{
  int step;
  int walltimeleft;
  real *box;
  real *boxhost;
  real *xhost; // DIM3*N_fixed coordinates followed by DIM3*N_free coordinates
  float *floatx; // xtc output use only
  real *shift;
  real *qhost;
  int *res_id;
  char (*res_name)[5];
  int *atom_id;
  char (*atom_name)[5];
  int N_all;
  int abortflag;
  struct struct_atoms *atoms;
  struct struct_atoms *Qdrive;
  real *Qdrivehost;
  real *bufhost;
  real *Gt;
  real *Gthost;
  real *Gts;
  real *Kt;
  real *Kthost;
  struct struct_bias *bias;
  struct struct_nblist *nblist_H4H;
  struct struct_nblist *nblist_H4D;
  cudaStream_t stream_random;
  cudaStream_t stream_bond;
  cudaStream_t stream_angle;
  cudaStream_t stream_dih;
  cudaStream_t stream_pair1;
  cudaStream_t stream_kinetic;
  cudaStream_t stream_nonbonded;
  // cudaStream_t stream_elec;
  cudaStream_t stream_update;
  cudaStream_t stream_default;
  cudaEvent_t event_bdone;
  cudaEvent_t event_adone;
  cudaEvent_t event_ddone;
  cudaEvent_t event_p1done;
  cudaEvent_t event_kdone;
  cudaEvent_t event_biasdone;
  cudaEvent_t event_nbdone;
  cudaEvent_t event_nbhdone;
  // cudaEvent_t event_esdone;
  cudaEvent_t event_fdone;
  cudaEvent_t event_updone;
} struct_state;

struct_state* alloc_state(struct_md* md);

void free_state(struct_state* state);

#endif

