#ifndef MD_RNA_ION_SPECIFIC_H
#define MD_RNA_ION_SPECIFIC_H

#include "defines.h"

#include "md.h"
#include "sort.h"
#include "topol.h"
#include "atoms.h"
#include "state.h"

void getforce_bond(struct_md* md);
void gethessian_bond(struct_md* md);

void getforce_angle(struct_md* md);
void gethessian_angle(struct_md* md);

void getforce_dih(struct_md* md);
void gethessian_dih(struct_md* md);

void getforce_pair1(struct_md* md);
void gethessian_pair1(struct_md* md);

void getforce_pair5(struct_md* md);

void getforce_pair6(struct_md* md);
void gethessian_pair6(struct_md* md);

void getforce_pair7(struct_md* md);

void getforce_pair8(struct_md* md);
void gethessian_pair8(struct_md* md);

void upload_bonded_d(
  int N_b,struct_bondparms* h_bondparms,struct_bondblock* h_bondblock,
  int N_a,struct_angleparms* h_angleparms,struct_angleblock* h_angleblock,
  int N_d,struct_dihparms* h_dihparms,struct_dihblock* h_dihblock,
  int N_p1,struct_pair1parms* h_pair1parms,
  int N_p6,struct_pair6parms* h_pair6parms,
  struct_atoms h_at,real* h_box,real* h_Gt);

__global__ void getforce_bonded_d();
__global__ void gethessian_bonded_d(struct_bias bias);

#endif
