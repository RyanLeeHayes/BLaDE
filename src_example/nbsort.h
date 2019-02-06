#ifndef MD_RNA_ION_SORT_H
#define MD_RNA_ION_SORT_H

#include "defines.h"

#include "nblist.h"

void quicksort_bins(struct_bin *bins,int i1,int i2);

void bitonic_sort(cudaStream_t strm,struct_nblist *H4D);

#endif
