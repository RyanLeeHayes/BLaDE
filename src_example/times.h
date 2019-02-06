#ifndef MD_RNA_IONS_TIMES_H
#define MD_RNA_IONS_TIMES_H

#include "defines.h"

#include "md.h"

typedef unsigned long long gmx_cycles_t;

typedef struct struct_times
{
  gmx_cycles_t start;
  gmx_cycles_t stop;
  gmx_cycles_t nsearch;
    gmx_cycles_t nblist_receive;
    gmx_cycles_t nblist_bin;
    gmx_cycles_t nblist_block;
    gmx_cycles_t nblist_tile;
    gmx_cycles_t nblist_check;
    gmx_cycles_t nblist_send;
  gmx_cycles_t force;
    gmx_cycles_t force_bond;
    gmx_cycles_t force_angle;
    gmx_cycles_t force_dih;
    gmx_cycles_t force_pair;
    gmx_cycles_t force_elec;
    gmx_cycles_t force_other;
    gmx_cycles_t force_sum;
  gmx_cycles_t update;
  gmx_cycles_t write;
  real secpcyc;
} struct_times;

struct_times* alloc_times(void);

void free_times(struct_times* times);

real gmx_cycles_calibrate(real sampletime);

void checkwalltime(struct_md* md,gmx_cycles_t start);

// Stealing gromacs counter
#if ((defined BGQ) || (defined BIOU))
static __inline__ gmx_cycles_t gmx_cycles_read(void)
{
    /* PowerPC using gcc inline assembly (and xlC>=7.0 with -qasm=gcc) */
    unsigned long low, high1, high2;
    do
    {
        __asm__ __volatile__ ("mftbu %0" : "=r" (high1) : );
        __asm__ __volatile__ ("mftb %0" : "=r" (low) : );
        __asm__ __volatile__ ("mftbu %0" : "=r" (high2) : );
    }
    while (high1 != high2);

    return (((gmx_cycles_t)high2) << 32) | (gmx_cycles_t)low;
}
#else
static __inline__ gmx_cycles_t gmx_cycles_read(void)
{
    /* x86 with GCC inline assembly - pentium TSC register */
    gmx_cycles_t   cycle;
    unsigned       low, high;

    __asm__ __volatile__("rdtsc" : "=a" (low), "=d" (high));

    cycle = ((unsigned long long)low) | (((unsigned long long)high)<<32);

    return cycle;
}
#endif

#endif

