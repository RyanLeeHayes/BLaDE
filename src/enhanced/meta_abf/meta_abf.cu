#include "enhanced/meta_abf/meta_abf.h"
#include "msld/msld.h"
#include "run/run.h"
#include "system/state.h"
#include "system/system.h"
#include "enhanced/enhanced.h"
#include "system/potential.h"
#include "main/gpu_check.h"
#include "main/real3.h"
#include "io/io.h"
#include <string>

MetaAdaptiveBiasingForce::~MetaAdaptiveBiasingForce(){
  if(counts) free(counts);
  if(counts_d) cudaFree(counts_d);
  if(dUdL_m2) free(dUdL_m2);
  if(dUdL_m2_d) cudaFree(dUdL_m2_d);
  if(dUdL_std) free(dUdL_std);
  if(dUdL_std_d) cudaFree(dUdL_std_d);
  if(dUdL_avg) free(dUdL_avg);
  if(dUdL_avg_d) cudaFree(dUdL_avg_d);
  if(meta_weights) free(meta_weights);
  if(meta_weights_d) cudaFree(meta_weights_d);
  if(meta_current_bias_d) cudaFree(meta_current_bias_d);
};

void parse_meta_abf(char* line, MetaAdaptiveBiasingForce* meta_abf){
  char token[MAXLENGTHSTRING];
  io_nexta(line, token);
  if(strcmp(token, "sample_freq") == 0){
    meta_abf->sample_freq=io_nexti(line);
  } else if (strcmp(token, "n_bins") == 0){
    if(!meta_abf->init){
      meta_abf->n_bins=io_nexti(line);
      if(meta_abf->n_bins<2){
        printf("Please choose a reasonable number of bins!\n");
        printf("Exiting...\n"); exit(1);
      }
    } else {
      printf("!!!!! Cannot change n_bins after initialization, leaving with %d bins!\n", meta_abf->n_bins);
    }
  } else if (strcmp(token, "target_site")==0){
    meta_abf->target_site=io_nextb(line);
  } else if (strcmp(token, "do_meta")==0){
    meta_abf->do_meta=io_nextb(line);
  } else if (strcmp(token, "do_abf")==0){
    meta_abf->do_abf=io_nextb(line);
  } else if (strcmp(token, "do_temper")==0){
    meta_abf->do_temper=io_nextb(line);
  } else if (strcmp(token, "do_sample")==0){
    meta_abf->do_sample=io_nextb(line);
  } else if (strcmp(token, "do_restart")==0){
    meta_abf->do_restart=io_nextb(line);
  } else if (strcmp(token, "abf_warmup")==0){
    meta_abf->abf_warmup=io_nexti(line);
  } else if (strcmp(token, "temper_factor")==0){
    meta_abf->temper_factor=io_nextf(line);
    if(meta_abf->temper_factor < 1.0 || abs(meta_abf->temper_factor-1)< 1e-4){
      printf("Temper factor too small!\n");
      exit(1);
    }
  } else if (strcmp(token, "meta_bias_mag") == 0){
    meta_abf->meta_bias_mag=io_nextf(line);
  } else if (strcmp(token, "meta_std")==0){
    meta_abf->meta_std = io_nextf(line);
  } else if (strcmp(token, "write_restart_freq")==0){
    meta_abf->write_restart_freq=io_nexti(line);
  } else if (strcmp(token, "log_freq")==0){
    meta_abf->log_freq=io_nexti(line);
  } else {
    printf("Didn't recognize option: %s\n", token);
    exit(1);
  }
};

// This only gets called the first time enhanced->initialize() gets called
void MetaAdaptiveBiasingForce::initialize(System* system){
    printf("Initializing Meta-ABF!\n");
    // ABF Memory
    counts = (real*)calloc(n_bins, sizeof(real));
    cudaMalloc(&counts_d, n_bins*sizeof(real));
    cudaMemcpy(counts_d, counts, n_bins*sizeof(real), cudaMemcpyDefault);
    dUdL_m2 = (real*)calloc(n_bins, sizeof(real));
    cudaMalloc(&dUdL_m2_d, n_bins*sizeof(real));
    cudaMemcpy(dUdL_m2_d, dUdL_m2, n_bins*sizeof(real), cudaMemcpyDefault);
    dUdL_std = (real*)calloc(n_bins, sizeof(real));
    cudaMalloc(&dUdL_std_d, n_bins*sizeof(real));
    cudaMemcpy(dUdL_std_d, dUdL_std, n_bins*sizeof(real), cudaMemcpyDefault);
    dUdL_avg = (real*)calloc(n_bins, sizeof(real));
    cudaMalloc(&dUdL_avg_d, n_bins*sizeof(real));
    cudaMemcpy(dUdL_avg_d, dUdL_avg, n_bins*sizeof(real), cudaMemcpyDefault);
    // Metadynamics Memory
    meta_weights = (real*)calloc(n_bins, sizeof(real));
    cudaMalloc(&meta_weights_d, n_bins*sizeof(real));
    cudaMemcpy(meta_weights_d, meta_weights, n_bins*sizeof(real), cudaMemcpyDefault);

    cudaMalloc(&meta_current_bias_d, sizeof(real));
    cudaMemcpy(meta_current_bias_d, &meta_current_bias, sizeof(real), cudaMemcpyDefault);

    // Read restart files
    if (do_restart) restart(system);

    if (target_site <= 0 || target_site >= system->msld->siteCount){
      printf("Choose a valid site! %d is not a valid site!\n", target_site);
      exit(1);
    }
    if (system->msld->blocksPerSite[target_site] != 2){
      printf("Meta ABF cannot be used on a site with more than 2 substituents!\n");
      exit(1);
    }

    // Update with current options
    bin_width = 1.0/(n_bins-1);
    half_search_bins = ceil(search_std*meta_std/bin_width);
    if (do_meta && half_search_bins > n_bins-1){
      printf("Requested search longer then 2 Lambda widths. Metadynamics reflecting boundary does not support this. Please reduce meta_std!\n");
      exit(1);
    }

    init=true;
};

int __host__ __device__ get_histogram_index(int n_bins, real x, real l, real u){
  // first part gets x progress through range [0, 1] 
  // bins centered on l and u have half width -> n_bins-1
  return (int)round(((x-l)/(u-l)) * (n_bins-1)); 
}

void __global__ getforce_abf_kernel(int n_bins, 
  real* lambda, real* dUdL_avg, real* counts, int abf_warmup,
  real* lambdaForce, real_e* energy){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real lEnergy=0;
  extern __shared__ real sEnergy[];
  if (i < n_bins){
    real L = 1.0-lambda[0];
    real bin_width = (1.0)/(n_bins-1.0);
    real dUdL_curr = dUdL_avg[i];
    dUdL_curr *= counts[i] < abf_warmup && abf_warmup > 0 ? counts[i]/abf_warmup : 1;
    if (i >= 1){ // Each thread computes integral from previous bin to this bin
      real dUdL_prev = dUdL_avg[i-1];
      dUdL_prev *= counts[i-1] < abf_warmup && abf_warmup > 0 ? counts[i-1]/abf_warmup : 1;
      lEnergy = bin_width*(dUdL_curr+dUdL_prev)/2.0; // trapezoid up to lambda
      if(L >= (i-1.0)*bin_width && L < i*bin_width){ // L is between last bin center and current bin center
        real interp = (L-(i-1.0)*bin_width)/bin_width;
        real dUdL_up = (1.0-interp)*dUdL_prev + interp*dUdL_curr;
        real width = (L-(i-1.0)*bin_width);
        lEnergy = width*(dUdL_prev+dUdL_up)/2.0;
        atomicAdd(&lambdaForce[0], dUdL_up); 
      } else if(L <= (i-1)*bin_width){ // L is less than lower bin center
        lEnergy = 0;
      }
    }
  }
  if (energy){
    // ABF adds -'ve F(L)
    real_sum_reduce(-lEnergy,sEnergy,energy);
  }
};

void __global__ getforce_meta_kernel(
  int n_bins, int n_search, real* lambda, 
  real meta_std, real* meta_weights, 
  bool update_current_only,
  real* lambdaForce, real* current_bias, real_e* energy){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  extern __shared__ real sEnergy[];
  real lEnergy=0;

  if (i < n_search){
    int half_search = (n_search-1)/2;
    real L = lambda[1];
    int bin = get_histogram_index(n_bins, L, 0, 1);
    real bin_width = 1.0/(n_bins-1.0);
    int my_bin = bin + (i-half_search);
    real my_bin_center = my_bin*bin_width; // don't update this with mirror
    if (my_bin < 0){ // lower mirror
      my_bin = -my_bin;
    } // mirror doesn't handle multiple reflections
    if (my_bin >= n_bins){ // upper mirror
      int overshoot = my_bin - (n_bins-1);
      my_bin = (n_bins-1) - overshoot; // max_id - overshoot
    }
    real dist = (L-my_bin_center)/meta_std;
    real gauss = exp(-.5*dist*dist);
    // first and last bins should have their weights doubled from contribution on other side of the mirror
    real mirror_factor = my_bin == 0 || my_bin == n_bins-1 ? 2.0 : 1.0;
    lEnergy = mirror_factor*meta_weights[my_bin]*gauss;
    real dUdL = -dist/meta_std*lEnergy; // need variance in denom
    if(lambdaForce && !update_current_only) atomicAdd(&lambdaForce[1], dUdL); 
  }
  if (energy && current_bias){
    if (!update_current_only) real_sum_reduce(lEnergy,sEnergy,energy);
    real_sum_reduce(lEnergy,sEnergy,current_bias);
  }
};

void getforce_meta_abf(System* system, int step, bool calcEnergy){
  MetaAdaptiveBiasingForce* m_abf = system->enhanced->meta_abf;
  State* state = system->state;
  Run* run = system->run;

  int shMem=0;
  real_e *pEnergy=NULL;
  if (calcEnergy) {
    shMem=BLBO*sizeof(real)/32;
    pEnergy=state->energy_d+eeenhanced;
  }

  int id = system->msld->siteBound[m_abf->target_site];
  if (m_abf->do_abf) {
    getforce_abf_kernel<<<(m_abf->n_bins+BLBO-1)/BLBO,BLBO,shMem,run->enhancedStream>>>(
      m_abf->n_bins, &state->lambda_fd[id], m_abf->dUdL_avg_d, m_abf->counts_d,m_abf->abf_warmup,
      &state->lambdaForce_d[id],pEnergy);
  }
  if (m_abf->do_meta) {
    int bins = 2*m_abf->half_search_bins + 1;
    cudaMemsetAsync(m_abf->meta_current_bias_d, 0, sizeof(real), run->enhancedStream);
    getforce_meta_kernel<<<(bins+BLBO-1)/BLBO,BLBO,shMem,run->enhancedStream>>>(
      m_abf->n_bins, bins, &state->lambda_fd[id], m_abf->meta_std, m_abf->meta_weights_d, 
      false, // update everything
      &state->lambdaForce_d[id], m_abf->meta_current_bias_d, pEnergy);
  }
  gpuCheck(cudaPeekAtLastError());
};

void __global__ add_sample_abf(
  int n_bins, real* lambda, real* lambdaForce, 
  real* counts, real* dUdL_m2, 
  real* dUdL_avg, real* dUdL_std){
    // stable online average and std - Welford's
    real L = 1-lambda[0]; // lambda = reference state lambda, 1-L means binning means int_0^1 gives dG 0->1
    int bin = get_histogram_index(n_bins, L, 0, 1);
    real dUdL = lambdaForce[1]-lambdaForce[0];
    counts[bin] += 1;
    real prev_delta = dUdL - dUdL_avg[bin];
    dUdL_avg[bin] += prev_delta/counts[bin];
    dUdL_m2[bin] += prev_delta*(dUdL-dUdL_avg[bin]);
    dUdL_std[bin] = sqrt(dUdL_m2[bin]/counts[bin]);
}

void __global__ add_sample_meta(int n_bins, real kT, bool do_temper,
  real* lambda, real bias_mag, real* current_bias, real temper_factor,
  real* meta_weights){
    real L = 1.0-lambda[0]; // lambda = reference state lambda, 1-L means binning means int_0^1 gives dG 0->1
    int bin = get_histogram_index(n_bins, L, 0.0, 1.0);
    real factor = do_temper ? exp(-current_bias[0]/((temper_factor-1.0)*kT)): 1.0;
    meta_weights[bin] += bias_mag*factor;
}

void sample_meta_abf(System* system, int step){
  MetaAdaptiveBiasingForce* m_abf = system->enhanced->meta_abf;
  State* state = system->state;
  Run* run = system->run;

  if(m_abf->do_sample && step % m_abf->sample_freq == 0){
    int id = system->msld->siteBound[m_abf->target_site];
    if(m_abf->do_abf){
      add_sample_abf<<<1, 1, 0, run->enhancedStream>>>(
        m_abf->n_bins, &state->lambda_fd[id], &state->lambdaForce_d[id],
        m_abf->counts_d, m_abf->dUdL_m2_d, 
        m_abf->dUdL_avg_d, m_abf->dUdL_std_d);
    }
    if(m_abf->do_meta){
      // Update current value of the potential
      int bins = 2*m_abf->half_search_bins + 1;
      int shMem=BLBO*sizeof(real)/32;
      real_e* pEnergy=state->energy_d+eeenhanced;
      cudaMemsetAsync(m_abf->meta_current_bias_d, 0, sizeof(real), run->enhancedStream);
      getforce_meta_kernel<<<(bins+BLBO-1)/BLBO,BLBO,shMem,run->enhancedStream>>>(
        m_abf->n_bins, bins, &state->lambda_fd[id], m_abf->meta_std, m_abf->meta_weights_d, 
        true, // only update meta_current_bias_d
        &state->lambdaForce_d[id], m_abf->meta_current_bias_d, pEnergy);
      // Add sample
      add_sample_meta<<<1, 1, 0, run->enhancedStream>>>(
        m_abf->n_bins, kB*run->T, m_abf->do_temper,
        &state->lambda_fd[id], 
        m_abf->meta_bias_mag, m_abf->meta_current_bias_d, m_abf->temper_factor,m_abf->meta_weights_d);
    }
  }
  gpuCheck(cudaPeekAtLastError());
};

void recv_meta_abf(System* system){
  MetaAdaptiveBiasingForce* m_abf = system->enhanced->meta_abf;
  cudaMemcpy(m_abf->counts, m_abf->counts_d, m_abf->n_bins*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(m_abf->dUdL_m2, m_abf->dUdL_m2_d, m_abf->n_bins*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(m_abf->dUdL_avg, m_abf->dUdL_avg_d, m_abf->n_bins*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(m_abf->dUdL_std, m_abf->dUdL_std_d, m_abf->n_bins*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(m_abf->meta_weights, m_abf->meta_weights_d, m_abf->n_bins*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(&m_abf->meta_current_bias, m_abf->meta_current_bias_d, sizeof(real), cudaMemcpyDefault);
};

void print_real_array(real* arr, int len){
  if(arr){
    printf("[ ");
    for(int i = 0; i < len; i++){
      if(i == len-1){
        printf("%7.2f ", arr[i]);
      } else {
        printf("%7.2f, ", arr[i]);
      }
    }
    printf("]");
  }
}

real eval_weights_at_point(real L, real* weights, real std, int bins, int search){
  int current_bin = get_histogram_index(bins, L, 0, 1);
  real U = 0;
  for(int i = current_bin-search; i <= current_bin+search; i++){
    int i_mirrored = i < 0 ? -i : i;
    i_mirrored = i >= bins ? 2*(bins-1)-i : i;
    real L_center = i*(1.0/(bins-1.0));
    real dist = (L-L_center)/std;
    U += weights[i_mirrored]*exp(-.5*dist*dist);
  }
  return U;
}

void log_meta_abf(System* system, int step){
  MetaAdaptiveBiasingForce* m_abf = system->enhanced->meta_abf;
  State* state = system->state;
  Msld* msld = system->msld;
  if(m_abf->log_freq != 0 && step % m_abf->log_freq == 0){
    state->recv_energy();
    if(!m_abf->do_sample){ printf("NOT ADDING SAMPLES!!!!\n");}
    printf("Step %d, U_enhanced: %8.2f:\n", step, system->state->energy[eeenhanced]);
    recv_meta_abf(system);
    state->recv_lambda();
    state->recv_position();
    cudaMemcpy(state->lambdaForce, state->lambdaForce_d, state->lambdaCount*sizeof(real), cudaMemcpyDefault);
    int block_start = msld->siteBound[m_abf->target_site];
    printf("Target Site: %d, Ref Block: %d, Lambda: %3.2f (%.4e), dUdL0: %8.2f, dUdL1: %8.2f, dUdL1-dUdL0: %8.2f\n", 
      m_abf->target_site, block_start, 1.0-state->lambda[block_start], 1.0-state->lambda[block_start], 
      state->lambdaForce[block_start], state->lambdaForce[block_start+1], 
      state->lambdaForce[block_start+1]-state->lambdaForce[block_start]
    );
    if (m_abf->do_meta && m_abf->do_temper) {
      real kT = kB*system->run->T;
      real factor = exp(-m_abf->meta_current_bias/(kT*(m_abf->temper_factor-1.0)));
      printf("Current Bias: %5.2f, Temper Factor: %5.2f,  Decay Factor: %5.2f\n", m_abf->meta_current_bias, m_abf->temper_factor, factor);
    }
    real dG_meta = 0;
    real dG_TI = 0;
    if (m_abf->do_meta){
      printf("Meta Weights: ");
      print_real_array(m_abf->meta_weights, m_abf->n_bins);
      printf("\n");
      dG_meta = eval_weights_at_point(1, m_abf->meta_weights, m_abf->meta_std, m_abf->n_bins, m_abf->half_search_bins);
      dG_meta -= eval_weights_at_point(0, m_abf->meta_weights, m_abf->meta_std, m_abf->n_bins, m_abf->half_search_bins);
    }
    if (m_abf->do_abf){
      printf("Counts: ");
      print_real_array(m_abf->counts, m_abf->n_bins);
      printf("\n");
      printf("<dU/dL>: ");
      print_real_array(m_abf->dUdL_avg, m_abf->n_bins);
      printf("\n");
      printf("std(dU/dL): ");
      print_real_array(m_abf->dUdL_std, m_abf->n_bins);
      printf("\n");
      real bin_width = 1.0/(m_abf->n_bins-1.0);
      for(int i = 0; i < m_abf->n_bins-1; i++){
        dG_TI += bin_width*(m_abf->dUdL_avg[i] + m_abf->dUdL_avg[i+1])/2.0;
      }
      // index zero is l0=1, index n_bins-1 is l0=0
    }
    printf("dG_{0->1,meta}: %f\n", dG_meta);
    printf("dG_{0->1,TI}: %f\n", dG_TI);
    printf("dG_{0->1,combined}: %f\n", dG_meta+dG_TI);
    printf("\n");
  }
};

// As usual, claude wrote the file write and reads
void write_meta_abf(std::string dir_name, System* system, int step){
  MetaAdaptiveBiasingForce* m_abf = system->enhanced->meta_abf;
  if (step % m_abf->write_restart_freq == 0){
    std::string filename = dir_name + "/" +  m_abf->fnm_meta_abf;
    recv_meta_abf(system);
    FILE* fp = fopen(filename.c_str(), "w");
    if(!fp){
      printf("Error: could not open %s for writing!\n", filename.c_str());
      printf("Exiting...\n"); exit(1);
    }
    /*
      File structure:
      target_site #
      n_bins #
      counts # # # # ...
      dUdL_m2 # # # # ...
      dUdL2_sum # # # # ...
      meta_weights # # # # ...
    */
    fprintf(fp, "target_site %d\n", m_abf->target_site);
    fprintf(fp, "n_bins %d\n",      m_abf->n_bins);

    fprintf(fp, "counts");
    for(int i = 0; i < m_abf->n_bins; i++){
      fprintf(fp, " %f", m_abf->counts[i]);
    }
    fprintf(fp, "\n");

    fprintf(fp, "dUdL_m2");
    for(int i = 0; i < m_abf->n_bins; i++){
      fprintf(fp, " %f", m_abf->dUdL_m2[i]);
    }
    fprintf(fp, "\n");

    fprintf(fp, "dUdL_avg");
    for(int i = 0; i < m_abf->n_bins; i++){
      fprintf(fp, " %f", m_abf->dUdL_avg[i]);
    }
    fprintf(fp, "\n");

    fprintf(fp, "meta_weights");
    for(int i = 0; i < m_abf->n_bins; i++){
      fprintf(fp, " %f", m_abf->meta_weights[i]);
    }
    fprintf(fp, "\n");
    fflush(fp);
    fclose(fp);
  }
};

void MetaAdaptiveBiasingForce::restart(System* system){
  std::string fnm = system->enhanced->output_dir + "/" + fnm_meta_abf;
  FILE* fp = fopen(fnm.c_str(), "r");
  if(!fp){
    printf("MetaABF: output dir (%s)\n", system->enhanced->output_dir.c_str());
    printf("MetaABF: no restart file found (%s), starting fresh.\n", fnm.c_str());
    return;
  }
  printf("MetaABF: reading restart file %s\n", fnm.c_str());

  char line[4096];
  char token[MAXLENGTHSTRING];
  while(fgets(line, sizeof(line), fp)){
    char* pos = line;
    int nread=0;
    if(sscanf(pos, "%s%n", token, &nread) != 1) continue;
    pos += nread;
    if(strcmp(token, "target_site") == 0){
      int read_target_site;
      sscanf(pos, " %d", &read_target_site);
      if(read_target_site != target_site){
        printf("Warning: restart target_site (%d) differs from current target_site (%d). Using current.\n",
               read_target_site, target_site);
      }
    } else if(strcmp(token, "n_bins") == 0){
      int read_n_bins;
      sscanf(pos, " %d", &read_n_bins);
      if(read_n_bins != n_bins){
        printf("Error: restart n_bins (%d) does not match current n_bins (%d)!\n",
               read_n_bins, n_bins);
        printf("Exiting...\n"); exit(1);
      }
    } else if(strcmp(token, "counts") == 0){
      for(int i = 0; i < n_bins; i++){
        nread = 0;
        int read = sscanf(pos, " %f%n", &counts[i], &nread);
        pos += nread;
      }
    } else if(strcmp(token, "dUdL_m2") == 0){
      for(int i = 0; i < n_bins; i++){
        int nread = 0;
        sscanf(pos, " %f%n", &dUdL_m2[i], &nread);
        pos += nread;
      }
    } else if(strcmp(token, "dUdL_avg") == 0){
      for(int i = 0; i < n_bins; i++){
        int nread = 0;
        sscanf(pos, " %f%n", &dUdL_avg[i], &nread);
        pos += nread;
      }
    } else if(strcmp(token, "meta_weights") == 0){
      for(int i = 0; i < n_bins; i++){
        int nread = 0;
        sscanf(pos, " %f%n", &meta_weights[i], &nread);
        pos += nread;
      }
    }
  }

  fclose(fp);

  // Compute std
  for(int i = 0; i < n_bins; i++){
    dUdL_std[i] = abs(counts[i]) > 1e-3 ? sqrt(dUdL_m2[i]/counts[i]) : 0;
  }

  // Update GPU memory with memory from file
  cudaMemcpy(counts_d, counts, n_bins*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(dUdL_m2_d, dUdL_m2, n_bins*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(dUdL_avg_d, dUdL_avg, n_bins*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(dUdL_std_d, dUdL_std, n_bins*sizeof(real), cudaMemcpyDefault);
  cudaMemcpy(meta_weights_d, meta_weights, n_bins*sizeof(real), cudaMemcpyDefault);

  printf("MetaABF: restart complete.\n");
};