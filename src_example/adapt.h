#ifndef MD_RNA_IONS_ADAPT_H
#define MD_RNA_IONS_ADAPT_H

#include "md.h"
#include "state.h"

void update_history(struct_md* md);

void get_metadynamics_bias(struct_md* md);

void read_metadynamics_bias(struct_bias* bias,struct_md* md);

void print_metadynamics_bias(struct_md* md);

#endif
