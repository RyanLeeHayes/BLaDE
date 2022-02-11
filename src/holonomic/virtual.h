#ifndef HOLONOMIC_VIRTUAL_H
#define HOLONOMIC_VIRTUAL_H

#include <cuda_runtime.h>

#include "main/defines.h"



// Forward definitions
class System;

void calc_virtual_position(System *system,bool rectify);
void calc_virtual_force(System *system);

#endif
