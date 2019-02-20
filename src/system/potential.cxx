#include "system/system.h"
#include "io/io.h"
#include "system/potential.h"

void parse_potential(char *line,System *system)
{
  fatal(__FILE__,__LINE__,"No parsing implemented for potential\n");
}

void Potential::calc_force(int step,System *system)
{
  // fprintf(stdout,"Force calculation placeholder (step %d)\n",step);
}
