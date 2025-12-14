#ifndef RESTRIAN_RESTRAIN_H
#define RESTRAINT_RESTRAIN_H

// Forward definitions
class System;

void getforce_noe(System *system,bool calcEnergy);
void getforce_harm(System *system,bool calcEnergy);
void getforce_boRest(System *system,bool calcEnergy);
void getforce_anRest(System *system,bool calcEnergy);
void getforce_diRest(System *system,bool calcEnergy);

#endif
