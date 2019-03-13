#include "main/defines.h"



// Forward declarations
class System;
struct NbondPotential; // system/potential.h

// Data structure used for sorting, sort by ix, then iy, then z.
struct DomdecBlockToken {
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
  real3 shift;
};

class Domdec {
  public:
  int3 gridDomdec;
  int3 idDomdec;
  int maxBlocks, maxPartnersPerBlock;
  int *domain_d;
  int localCount;
  int globalCount;
  int *localToGlobal_d;
  int *globalToLocal_d;
  real3 *localPosition_d;
  real3 *localForce_d;
  struct NbondPotential *localNbonds_d;
  struct DomdecBlockSort *blockSort_d;
  struct DomdecBlockToken *blockToken_d;
  int *blockBounds_d;
  int blockCount, *blockCount_d;
  struct DomdecBlockVolume *blockVolume_d;
  int *blockPartnerCount_d;
  struct DomdecBlockPartners *blockPartners_d;

  Domdec();
  ~Domdec();

  void initialize(System *system);
  void reset_domdec(System *system);

  // From domdec/assign_domain.cu
  void assign_domain(System *system);
  // From domdec/assign_block.cu
  void assign_blocks(System *system);
  void pack_positions(System *system);
  // From domdec/cull.cu
  void cull_blocks(System *system);
};
