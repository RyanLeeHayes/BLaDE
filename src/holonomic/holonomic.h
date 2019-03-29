#ifndef HOLONOMIC_HOLONOMIC_H
#define HOLONOMIC_HOLONOMIC_H

#include <cuda_runtime.h>

#include "main/defines.h"



// Forward definitions
class System;
class LeapState;

void holonomic_velocity(System *system);
void holonomic_position(System *system);
void holonomic_backup_position(LeapState *leapState,real *positionCons,cudaStream_t stream);

#endif
