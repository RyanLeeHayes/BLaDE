#ifndef UPDATE_UPDATE_H
#define UPDATE_UPDATE_H

// See https://aip.scitation.org/doi/abs/10.1063/1.1420460 for barostat

// Forward declarations
class System;

class Update {
  public:

  Update()
  {
  }

  ~Update()
  {
  }

  void initialize(System *system);
  void update(int step,System *system);
};

#endif
