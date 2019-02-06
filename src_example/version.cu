// Private version V42.4 = Public version 1.0
// Private version V42.16 = Public version 1.1

#include <stdio.h> 

#include "version.h"

void printversion(FILE *fp)
{
  fprintf(fp,"%s","Version 0.1.1\n"
    "Updated 2017-11-13 13:30\n");
  fprintf(fp,
    "Change log:\n"
    "  V0.1.1 - Branched from code for OSRW in Q on SMOG models\n");
}
