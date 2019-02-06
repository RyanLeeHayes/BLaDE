
#include "md.h"

#include "defines.h"

#include "parms.h"
#include "state.h"
#include "times.h"
#include "files.h"
#include "topol.h"

#include <stdlib.h>


struct_md* alloc_md(int argc, char *argv[])
{
  struct_md *md;
  md=(struct_md*) malloc(sizeof(struct_md));
  md->log=stdout;
  fprintf(md->log,"WARNING: need to set up a more reasonable log file on %s %d\n",__FILE__,__LINE__);
  md->parms=alloc_parms(argc,argv,md->log);
  md->state=alloc_state(md);
  md->parms=alloc_topparms(md);
  md->files=alloc_files();
  md->times=alloc_times();

  return md;
}


void free_md(struct_md* md)
{
  free_topparms(md->parms);
  free_parms(md->parms);
  free_state(md->state);
  free_files(md->files);
  free_times(md->times);
  free(md);
}
