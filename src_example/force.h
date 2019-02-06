#ifndef MD_RNA_IONS_FORCE_H
#define MD_RNA_IONS_FORCE_H

#include "defines.h"

#include "md.h"

#include "defines.h"

__global__ void resetforce_d(int N,real* f);

void get_FQ(struct_md* md);

void resetforce(struct_md* md);
void resetenergy(struct_md* md);
void resetbias(struct_md* md);

void getforce(struct_md* md);
void gethessian(struct_md* md);

#endif

