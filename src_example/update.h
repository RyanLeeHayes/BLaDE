#ifndef MD_RNA_IONS_RUNMD_H
#define MD_RNA_IONS_RUNMD_H

#include "defines.h"

#include "md.h"
#include "atoms.h"
#include "leapparms.h"
#include "mersenne.h"

__global__ void leapfrog(real* nu,struct_atoms at,struct_leapparms lp,real* box);
// void leapfrog(struct_md* md,struct_atoms* atoms,struct_leapparms* leapparms);

void update(struct_md* md);

void resetstep(struct_md* md);

void resetcumcycles(struct_md* md);

void upload_update_d(struct_atoms h_at);

#endif
