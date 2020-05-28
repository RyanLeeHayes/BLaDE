#include "main/defines.h"



// Forward declarations
class System;
struct NbondPotential; // system/potential.h
struct ExclPotential; // system/potential.h

// Data structure used for sorting, sort by ix, then iy, then z.
struct DomdecBlockToken {
  int domain;
  int ix,iy;
  real z;
};

// Structure for tree sort
// Tree structure and unsorted array have the same elements in the same order.
// Place the root of the whole tree at the end of the array
struct DomdecBlockSort {
  int root; // Index within the tree of the root/parent to this leaf
  int lower; // Index within the tree structure of the lower leaf/child
  int lowerCount; // Number of leaves in lower subtree
  int upper; // Index within the tree structure of the upper leaf/child
  int upperCount; // Number of leaves in upper subtree
  int whoCounts; // Which branch counts for sorting purposes -1 no one yet, 0 lower, 1 higher
};

struct DomdecBlockVolume {
  real3 min;
  real3 max;
};

struct DomdecBlockPartners {
  int jBlock;
  char4 shift;
  int exclAddress; // Tells where to look up exclusions
};

class Domdec {
  public:
  int id;
  int idCount;
  int3 gridDomdec;
  int3 idDomdec;
// Heuristic settings
  int maxBlocks, maxPartnersPerBlock;
  int freqDomdec;
  real cullPad;
  int maxBlockExclCount;
// Assign atoms to domains
  int *domain;
  int *domain_d;
  int *domain_omp;
// For converting between local and global indices
  int globalCount;
  int *localToGlobal_d;
  int *globalToLocal_d;
  real3 *localPosition_d;
  real3_f *localForce_d;
  struct NbondPotential *localNbonds_d;
// For sorting atoms into blocks
  struct DomdecBlockSort *blockSort_d;
  struct DomdecBlockToken *blockToken_d;
  int *blockBounds_d;
  int *blockCount, *blockCount_d;
// For deciding what's interacting
  struct DomdecBlockVolume *blockVolume_d;
  int *blockCandidateCount_d; // Candidates might interact and are defined each search
  struct DomdecBlockPartners *blockCandidates_d;
  int *blockPartnerCount_d; // Partners do interact and are defined each step
  struct DomdecBlockPartners *blockPartners_d;
// For exclusions
  struct ExclPotential *localExcls_d;
  struct DomdecBlockSort *exclSort_d;
  struct ExclPotential *sortedExcls_d;
#ifdef USE_TEXTURE
  cudaTextureObject_t sortedExcls_tex;
#endif
  int sortedExclCount;
  int *blockExcls_d;
  int *blockExclCount_d;

  Domdec();
  ~Domdec();

  void initialize(System *system);
  void reset_domdec(System *system);
  void update_domdec(System *system,bool resetFlag);

  // From domdec/assign_domain.cu
  void broadcast_domain(System *system);
  void assign_domain(System *system);
  // From domdec/assign_block.cu
  void assign_blocks(System *system);
  void pack_positions(System *system);
  void unpack_forces(System *system);
  // From domdec/cull.cu
  void cull_blocks(System *system);
  // From domdec/assign_excl.cu
  void setup_exclusions(System *system);
  // From domdec/recull.cu
  void recull_blocks(System *system);
};
