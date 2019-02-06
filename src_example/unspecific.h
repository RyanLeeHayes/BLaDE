#ifndef MD_RNA_IONS_UNSPECIFIC_H
#define MD_RNA_IONS_UNSPECIFIC_H

#include "defines.h"

#include "md.h"
#include "atoms.h"
#include "nblist.h"
#include "topol.h"

void getforce_other(struct_md* md);
void gethessian_other(struct_md* md);

void upload_other_d(real h_k12,real h_rc2,struct_atoms h_at,struct_nblist h_H4D,real* h_Gt);

void upload_kbgo_d(struct_nbparms *h_nbparms);

#endif

