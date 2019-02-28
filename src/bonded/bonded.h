#ifndef BONDED_BONDED_H
#define BONDED_BONDED_H

// Forward definitions
class System;

void getforce_bond(System *system,bool calcEnergy);
void getforce_angle(System *system,bool calcEnergy);
void getforce_dihe(System *system,bool calcEnergy);
void getforce_impr(System *system,bool calcEnergy);
void getforce_cmap(System *system,bool calcEnergy);

#endif
