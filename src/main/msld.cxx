#include "system/system.h"
#include "io/io.h"

int main(int argc, char *argv[])
{
  System *system;
  FILE *fp;

  system=new(System);

  // open input file
  if (argc < 2) {
    fatal(__FILE__,__LINE__,"Error: need input file\n");
  }
  fp=fpopen(argv[1],"r");

  interpretter(fp,system,1);

  fclose(fp);

  delete(system);
}
