#ifndef MAIN_DEFINES_H
#define MAIN_DEFINES_H

#include <cuda_runtime.h>
#include <cufft.h>

#define USE_TEXTURE

#define MAXLENGTHSTRING 1024

// Units
// At least this is uncontroversial...
#define DEGREES 0.017453292519943

#ifdef CHARMM_UNITS

// Use CHARMM units, energy: kcal/mol, distance: A, time: A/sqrt(kcal/mol/amu), mass: amu

// See values in charmm/source/ltm/consta_ltm.F90
#define ANGSTROM 1.0
#define FOURANGSTROM 4.0f
#define KCAL_MOL 1.0
// 1 / CHARMM TIMFAC value 0.0488882129
#define PICOSECOND 20.454828284385908
// CHARMM KBOLTZ
#define kB 0.001987191
// Charmm CCELEC0 value
#define kELECTRIC 332.0716f

// from gromacs values
#define ATMOSPHERE 0.000014584319218586

// NOTE kB and KCAL_MOL disagree by 3.01E-5 in 1 over temperature scale
// That's 0.01 K at T=300 K

#else

// Use gromacs units, energy: kJ/mol, distance: nm, time: ps, mass: amu

// 2019-02-28 Google searches
// eps0 8.85418782E-12 J*m/C^2
// e 1.60217662E-19 C
// NA 6.02214076E23
// ATM 101325 bar

#define ANGSTROM 0.1
#define FOURANGSTROM 0.4f
// 4.184 according to CHARMM, this number is from kELECTRIC conversion
#define KCAL_MOL 4.183900553475091
#define PICOSECOND 1.0
#define kB 0.0083144598
#define kELECTRIC 138.935455103336f

// atm -> bar -> kJ/mol/nm^3
#define ATMOSPHERE 0.0610193412507

// NOTE kB and KCAL_MOL disagree by 3.01E-5 in 1 over temperature scale
// That's 0.01 K at T=300 K

#endif

// CUDA block size for system update kernels
#define BLUP 256
#define BLBO 256
#define BLNB 256
#define BLMS 256

#ifdef DOUBLE
typedef double real;
typedef double4 real4;
typedef double3 real3;
typedef double2 real2;
typedef cufftDoubleReal myCufftReal;
typedef cufftDoubleComplex myCufftComplex;
#define myCufftExecR2C cufftExecD2Z
#define myCufftExecC2R cufftExecZ2D
#define MYCUFFT_R2C CUFFT_D2Z
#define MYCUFFT_C2R CUFFT_Z2D
#define MPI_CREAL MPI_DOUBLE
#else
typedef float real;
typedef float4 real4;
typedef float3 real3;
typedef float2 real2;
typedef cufftReal myCufftReal;
typedef cufftComplex myCufftComplex;
#define myCufftExecR2C cufftExecR2C
#define myCufftExecC2R cufftExecC2R
#define MYCUFFT_R2C CUFFT_R2C
#define MYCUFFT_C2R CUFFT_C2R
#define MPI_CREAL MPI_FLOAT
#endif

struct Int2 {
  int i[2];
};

struct Int3 {
  int i[3];
};

struct Int4 {
  int i[4];
};

struct Int8 {
  int i[8];
};

struct Real3 {
  real i[3];
};

#endif
