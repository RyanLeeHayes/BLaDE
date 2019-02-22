#ifndef RNG_RNG_CPU_H
#define RNG_RNG_CPU_H

#include "main/defines.h"

struct MTState
{
  unsigned long *mt; /* the array for the state vector  */
  int mti; /* mti==N+1 means mt[N] is not initialized */
  real nextReal;
  bool bNextReal;
};

class RngCPU
{
  public:
  struct MTState *mtState;

  RngCPU();
  ~RngCPU();

  void init_genrand(unsigned long s, unsigned long *mt, int* pt_mti);
  void init_by_array(unsigned long *init_key, int key_length, unsigned long *mt, int* pt_mti);
  unsigned long genrand_int32(unsigned long *mt,int* pt_mti);

  struct MTState* alloc_mtstate(unsigned long s);
  void free_mtstate(struct MTState* mts);
  real rand_normal();
  real rand_uniform();
};

#endif

