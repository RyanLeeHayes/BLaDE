#ifndef SYSTEM_STATE_H
#define SYSTEM_STATE_H

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#include <string>
#include <map>

#include "rng/rng_cpu.h"
#include "rng/rng_gpu.h"

// Forward delcarations
class System;

struct AtomState {
  std::string segName;
  int resIdx;
  std::string atomName;
};
bool operator<(const struct AtomState& a,const struct AtomState& b);
bool operator==(const struct AtomState& a,const struct AtomState& b);

class State {
  public:
  std::map<std::string,void(State::*)(char*,char*,System*)> parseState;
  std::map<std::string,std::string> helpState;

  std::map<struct AtomState,Real3> fileData;

  RngCPU *rngCPU;
  RngGPU *rngGPU;

  int atomCount;
  real box[3][3];
  real (*position)[3];
  float (*fposition)[3]; // Intentional float
  real (*velocity)[3];
  real (*force)[3];
  real (*mass)[3];
  real (*invsqrtMass)[3];
// Device versions
  real (*position_d)[3];
  real (*velocity_d)[3];
  real (*force_d)[3];
  real (*mass_d)[3];
  real (*invsqrtMass_d)[3];
  real (*random_d)[3];

  State(int n) {
    atomCount=n;
    box[0][0]=NAN;
    position=(real(*)[3])calloc(n,sizeof(real[3]));
#ifdef DOUBLE
    fposition=(float(*)[3])calloc(n,sizoef(float[3]));
#else
    fposition=position;
#endif
    velocity=(real(*)[3])calloc(n,sizeof(real[3]));
    force=(real(*)[3])calloc(n,sizeof(real[3]));
    mass=(real(*)[3])calloc(n,sizeof(real[3]));
    invsqrtMass=(real(*)[3])calloc(n,sizeof(real[3]));

    cudaMalloc(&(position_d),n*sizeof(real[3]));
    cudaMalloc(&(velocity_d),n*sizeof(real[3]));
    cudaMalloc(&(force_d),n*sizeof(real[3]));
    cudaMalloc(&(mass_d),n*sizeof(real[3]));
    cudaMalloc(&(invsqrtMass_d),n*sizeof(real[3]));
    cudaMalloc(&(random_d),2*n*sizeof(real[3]));

    rngCPU=new RngCPU;
    rngGPU=new RngGPU;

    setup_parse_state();
    fprintf(stdout,"IMPLEMENT State create %s %d\n",__FILE__,__LINE__);
  }

  ~State() {
    if (position) free(position);
#ifdef DOUBLE
    if (fposition) free(fposition);
#endif
    if (velocity) free(velocity);
    if (force) free(force);
    if (mass) free(mass);
    if (invsqrtMass) free(invsqrtMass);

    if (position_d) cudaFree(position_d);
    if (velocity_d) cudaFree(velocity_d);
    if (force_d) cudaFree(force_d);
    if (mass_d) cudaFree(mass_d);
    if (invsqrtMass_d) cudaFree(invsqrtMass_d);
    if (random_d) cudaFree(random_d);

    delete rngCPU;
    delete rngGPU;
  }

  void setup_parse_state();

  void help(char *line,char *token,System *system);
  void error(char *line,char *token,System *system);
  void reset(char *line,char *token,System *system);
  void file(char *line,char *token,System *system);
  void parse_box(char *line,char *token,System *system);
  void parse_velocity(char *line,char *token,System *system);
  void dump(char *line,char *token,System *system);

  void file_pdb(FILE *fp,System *system);

  void send_position() {
    cudaMemcpy(position_d,position,atomCount*sizeof(real[3]),cudaMemcpyHostToDevice);
  }
  void recv_position() {
    cudaMemcpy(position,position_d,atomCount*sizeof(real[3]),cudaMemcpyDeviceToHost);
  }
  void send_velocity() {
    cudaMemcpy(velocity_d,velocity,atomCount*sizeof(real[3]),cudaMemcpyHostToDevice);
  }
  void recv_velocity() {
    cudaMemcpy(velocity,velocity_d,atomCount*sizeof(real[3]),cudaMemcpyDeviceToHost);
  }
  void send_invsqrtMass() {
    cudaMemcpy(invsqrtMass_d,invsqrtMass,atomCount*sizeof(real[3]),cudaMemcpyHostToDevice);
  }
  void recv_invsqrtMass() {
    cudaMemcpy(invsqrtMass,invsqrtMass_d,atomCount*sizeof(real[3]),cudaMemcpyDeviceToHost);
  }
};

void parse_state(char *line,System *system);

#endif
