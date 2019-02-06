
#include "times.h"

#include "defines.h"

#include "md.h"
#include "parms.h"
#include "state.h"

#include <stdlib.h>
// #include <omp.h>
#include <sys/time.h>

struct_times* alloc_times(void)
{
  struct_times *times;

  times=(struct_times*) malloc(sizeof(struct_times));
  times->start=0;
  times->stop=0;
  times->nsearch=0;
    times->nblist_receive=0;
    times->nblist_bin=0;
    times->nblist_block=0;
    times->nblist_tile=0;
    times->nblist_check=0;
    times->nblist_send=0;
  times->force=0;
    times->force_bond=0;
    times->force_angle=0;
    times->force_dih=0;
    times->force_pair=0;
    times->force_elec=0;
    times->force_other=0;
    times->force_sum=0;
  times->update=0;
  times->write=0;
  return times;
}


void free_times(struct_times* times)
{
  free(times);
}


real gmx_cycles_calibrate(real sampletime)
{
    /*  generic implementation with gettimeofday() */
    struct timeval t1, t2;
    gmx_cycles_t   c1, c2;
    real         timediff, cyclediff;
    real         d = 0.1; /* Dummy variable so we don't optimize away delay lo
op */
    int            i;

    /* Start a timing loop. We want this to be largely independent
     * of machine speed, so we need to start with a very small number
     * of iterations and repeat it until we reach the requested time.
     *
     * We call gettimeofday an extra time at the start to avoid cache misses.
     */
    gettimeofday(&t1, NULL);
    gettimeofday(&t1, NULL);
    c1 = gmx_cycles_read();

    do
    {
        /* just a delay loop. To avoid optimizing it away, we calculate a number
         * that will underflow to zero in most cases. By conditionally adding it
         * to a result at the end it cannot be removed. n=10000 is arbitrary...
         */
        for (i = 0; i < 10000; i++)
        {
            d = d/(1.0+(real)i);
        }
        /* Read the time again */
        gettimeofday(&t2, NULL);
        c2       = gmx_cycles_read();
        timediff = (real)(t2.tv_sec-t1.tv_sec)+
            (real)(t2.tv_usec-t1.tv_usec)*1e-6;
    }
    while (timediff < sampletime);

    cyclediff = c2-c1;

    /* Add a very small result so the delay loop cannot be optimized away */
    if (d < 1e-30)
    {
        timediff += d;
    }

    /* Return seconds per cycle */
    return timediff/cyclediff;
}


void checkwalltime(struct_md* md,gmx_cycles_t start)
{
  // #pragma omp barrier
  // #pragma omp master
  {
    if (gmx_cycles_read()-start > md->parms->maxc) {
      md->state->walltimeleft=0;
    }
  }
  // #pragma omp barrier
}

