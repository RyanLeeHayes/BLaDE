#ifndef DOMDEC_HBONDLIST_H
#define DOMDEC_HBONDLIST_H

#include "main/defines.h"
#include "hbond/hbond.h"

class System;

// Default list capacity per donor. Overflow is detected and reported, never silent.
#define HBONDLIST_PER_DONOR 64

void hbondlist_update(System *system,bool checkOverflow);

#endif
