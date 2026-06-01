#ifndef META_ABF_H
#define META_ABF_H

#include "main/defines.h"
#include <string>

class System;

/*
  Transition Tempered Metadynamics + Adaptive Biasing Force along a single lambda dimension
  Supports calculations with multiple sites, but the WTM+ABF can only be applied on a single site with 2 substituents
*/
class MetaAdaptiveBiasingForce {
  public:
    MetaAdaptiveBiasingForce(){};
    ~MetaAdaptiveBiasingForce();
    void initialize(System* system);
    void restart(System* system);

    bool init=false;
    // Options
    bool do_abf=true;
    bool do_meta=true;
    bool do_temper=true;
    bool do_sample=true; 
    bool do_restart=true;
    int target_site=1; // must have two substitiuents at this site
    int total_samples=0;
    int n_bins=201; // number of bins
    int abf_warmup=500; // number of samples before full activation
    int sample_freq=10; // 1/step
    real meta_bias_mag=0.005; 
    real meta_std=0.02;
    real temper_factor=10; // units of kT
    // Not optional
    real search_std = 6; // look this many std for gaussians
    real bin_width = 1.0/(n_bins-1);
    int half_search_bins = ceil(search_std*meta_std/bin_width); // in each direction
    // ABF Memory
    real* counts=NULL;
    real* counts_d=NULL;
    real* dUdL_m2=NULL;
    real* dUdL_m2_d=NULL;
    real* dUdL_std=NULL;
    real* dUdL_std_d=NULL;
    real* dUdL_avg=NULL;
    real* dUdL_avg_d=NULL;
    // Metadynamics Memory
    real meta_current_bias = 0;
    real* meta_current_bias_d;
    real* meta_weights=NULL;
    real* meta_weights_d=NULL;
    // Restart 
    int write_restart_freq=1000;
    std::string fnm_meta_abf = "meta_abf.rst";
    FILE* fp_meta_abf = NULL;
    // Logging
    int log_freq=0;
};

void parse_meta_abf(char* line, MetaAdaptiveBiasingForce* meta_abf);
void getforce_meta_abf(System* system, int step, bool calcEnergy);
void sample_meta_abf(System* system, int step);
void log_meta_abf(System* system, int step);
void recv_meta_abf(System* system);
void write_meta_abf(std::string dir_name, System* system, int step);

#endif