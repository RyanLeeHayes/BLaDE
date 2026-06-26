#include <omp.h>
#include <string.h>
#include <cuda_runtime.h>

#include "io/io.h"
#include "system/system.h"
#include "system/selections.h"
#include "system/structure.h"
#include "system/state.h"
#include "system/potential.h"
#include "run/run.h"

#include "enhanced/enhanced.h"
#include "enhanced/meta_abf/meta_abf.h"
#include "main/gpu_check.h"

Enhanced::~Enhanced(){
  if(atom_selection_primary) free(atom_selection_primary);
  if(atom_selection_primary_d) cudaFree(atom_selection_primary_d);
  if(meta_abf) delete meta_abf;
}

void parse_enhanced(char* line, System* system){
  char token[MAXLENGTHSTRING];

  if (system->structure==NULL) {
    fatal(__FILE__,__LINE__,"enhanced selections cannot be defined until structure has been defined\n");
  }

  if (system->enhanced==NULL) {
    printf("Instantiating Enhanced class!\n");
    system->enhanced=new Enhanced();
  }
  Enhanced* nhcd = system->enhanced;

  io_nexta(line,token);
  nhcd->active = true;
  if (strcmp(token, "updating")==0){
    nhcd->updating = io_nextb(line);
  } else if (strcmp(token,"dnmo")==0){
    nhcd->output_dir = io_nexts(line);
  } else if (strcmp(token,"atom_selection")==0){
    std::string name=io_nexts(line);
    if (system->selections->selectionMap.count(name)==0) {
      fatal(__FILE__,__LINE__,"Selection %s not found\n",name.c_str());
    }
    nhcd->primary_sele=name;
    if (nhcd->atom_selection_primary) free(nhcd->atom_selection_primary);
    nhcd->atom_selection_primary=(int*)calloc(system->structure->atomList.size(),sizeof(int));
    for (int i=0; i<system->structure->atomList.size(); i++) {
      nhcd->atom_selection_primary[i]=system->selections->selectionMap[name].boolSelection[i];
    }
    cudaMalloc(&nhcd->atom_selection_primary_d, system->structure->atomList.size()*sizeof(int));
    cudaMemcpy(nhcd->atom_selection_primary_d, nhcd->atom_selection_primary, system->structure->atomList.size()*sizeof(int), cudaMemcpyDefault);
  } else if (strcmp(token,"print_sele")==0){
    printf("Dumping enhanced selections! Will segfault if no selection is defined.\n");
    printf("Primary Selection Name: %s\n", nhcd->primary_sele.c_str());
    system->selections->dump();
  } else if (strcmp(token, "nbrecip_mode")==0){
    int sel = io_nexti(line);
    if(sel < 0 || sel > 2){
      printf("Only 0-2 nbrecip_mode supported {correct, on, off}!\n");
      exit(1);
    }
    nhcd->nbrecip_mode=sel;
  } else if (strcmp(token, "meta_abf") == 0){ // init class
    if (!nhcd->meta_abf){
      printf("Instantiating Meta-ABF!\n");
      nhcd->meta_abf = new MetaAdaptiveBiasingForce();
    } else {
      printf("Already instantiated Meta-ABF, reusing exising one!\n");
    }
  } else if (strcmp(token, "meta_abf_option") == 0){ // see meta_abf.cu for availible options
    if(!nhcd->meta_abf){
      printf("Meta_ABF not defined yet!\n");
      exit(1);
    }
    parse_meta_abf(line, nhcd->meta_abf);
  } else {
    printf("Didn't recognize option %s\n", token);
    exit(1);
  }
};

// Gets called each time a new "run" function is called like "run dynamics" or "run minimize"
void Enhanced::initialize(System* system){
  printf("Initializing Enhanced Class!\n");
  if(meta_abf && !meta_abf->init) meta_abf->initialize(system); 
  init = true; 
}

void getforce_enhanced(System* system, int step, bool calcEnergy){
  Enhanced* nhcd = system->enhanced;
  if (system->run->calcTermFlag[eeenhanced]==false) return;
  bool pressure = step == 0;
  if(nhcd->meta_abf){
    // internal sample/write/log frequency checks
    if (!pressure) { // sampling needs to be first
      sample_meta_abf(system, step); 
    }
    getforce_meta_abf(system, step, calcEnergy);
    if (!pressure) {
      write_meta_abf(nhcd->output_dir, system, step); 
      log_meta_abf(system, step);
    };
  }
}