#ifndef MAIN_DEFINES_H
#define MAIN_DEFINES_H

#include <cuda_runtime.h>
#include <cufft.h>

#ifndef DOUBLE
#define USE_TEXTURE
#endif

// #define REPLICAEXCHANGE

#define MAXLENGTHSTRING 1024

// Units
// At least this is uncontroversial...
#define DEGREES ((real)0.017453292519943)

// If this is a CHARMM compilation
#ifdef BLADE_IN_CHARMM
#define CHARMM_UNITS
#endif

#ifdef CHARMM_UNITS

// Use CHARMM units, energy: kcal/mol, distance: A, time: A/sqrt(kcal/mol/amu), mass: amu

// See values in charmm/source/ltm/consta_ltm.F90
#define ANGSTROM ((real)1.0)
#define SOFTCORERADIUS ((real)4.0)
#define KCAL_MOL ((real)1.0)
// 1 / CHARMM TIMFAC value 0.0488882129
#define PICOSECOND ((real)20.454828284385908)
// CHARMM KBOLTZ
#define kB ((real)0.001987191)
// Charmm CCELEC0 value
#define kELECTRIC ((real)332.0716)

// from gromacs values
#define ATMOSPHERE ((real)0.000014584319218586)

// NOTE kB and KCAL_MOL disagree by 3.01E-5 in 1 over temperature scale
// That's 0.01 K at T=300 K

#else

// Use gromacs units, energy: kJ/mol, distance: nm, time: ps, mass: amu

// 2019-02-28 Google searches
// eps0 8.85418782E-12 J*m/C^2
// e 1.60217662E-19 C
// NA 6.02214076E23
// ATM 101325 bar

#define ANGSTROM ((real)0.1)
#define SOFTCORERADIUS ((real)0.4)
// 4.184 according to CHARMM, this number is from kELECTRIC conversion
#define KCAL_MOL ((real)4.183900553475091)
#define PICOSECOND ((real)1.0)
#define kB ((real)0.0083144598)
#define kELECTRIC ((real)138.935455103336)

// atm -> bar -> kJ/mol/nm^3
#define ATMOSPHERE ((real)0.0610193412507)

// NOTE kB and KCAL_MOL disagree by 3.01E-5 in 1 over temperature scale
// That's 0.01 K at T=300 K

#endif

// CUDA block size for system update kernels
#define BLUP 256
#define BLBO 256
#define BLNB 256
#define BLMS 256

#define DOUBLE_E
// #define DOUBLE_F
#define DOUBLE_X
// #define DOUBLE_V

#ifdef DOUBLE
typedef double real;
typedef double4 real4;
typedef double3 real3;
typedef double2 real2;
typedef double1 real1;
typedef cufftDoubleReal myCufftReal;
typedef cufftDoubleComplex myCufftComplex;
#define myCufftExecR2C cufftExecD2Z
#define myCufftExecC2R cufftExecZ2D
#define MYCUFFT_R2C CUFFT_D2Z
#define MYCUFFT_C2R CUFFT_Z2D
#ifdef REPLICAEXCHANGE
#define MYMPI_REAL MPI_DOUBLE
#endif
#else
typedef float real;
typedef float4 real4;
typedef float3 real3;
typedef float2 real2;
typedef float1 real1;
typedef cufftReal myCufftReal;
typedef cufftComplex myCufftComplex;
#define myCufftExecR2C cufftExecR2C
#define myCufftExecC2R cufftExecC2R
#define MYCUFFT_R2C CUFFT_R2C
#define MYCUFFT_C2R CUFFT_C2R
#ifdef REPLICAEXCHANGE
#define MYMPI_REAL MPI_FLOAT
#endif
#endif

// make_real3 helper to match make_float3/make_double3
#ifdef DOUBLE
#define make_real3 make_double3
#else
#define make_real3 make_float3
#endif

typedef struct real33 {real3 a; real3 b;} real33;
typedef struct real321 {real3 a; real2 b; real1 c;} real321;
typedef struct real123 {real1 a; real2 b; real3 c;} real123;

#ifdef DOUBLE_E
typedef double real_e;
#ifdef REPLICAEXCHANGE
#define MYMPI_REAL_E MPI_DOUBLE
#endif
#else
typedef real real_e;
#ifdef REPLICAEXCHANGE
#define MYMPI_REAL_E MYMPI_REAL
#endif
#endif

#ifdef DOUBLE_F
typedef double real_f;
typedef double3 real3_f;
#else
typedef real real_f;
typedef real3 real3_f;
#endif

#ifdef DOUBLE_X
typedef double real_x;
typedef double3 real3_x;
typedef double2 real2_x;
typedef double1 real1_x;
#define MYMPI_REAL_X MPI_DOUBLE
#else
typedef real real_x;
typedef real3 real3_x;
typedef real2 real2_x;
typedef real1 real1_x;
#define MYMPI_REAL_X MYMPI_REAL
#endif
typedef struct real33_x {real3_x a; real3_x b;} real33_x;
typedef struct real321_x {real3_x a; real2_x b; real1_x c;} real321_x;
typedef struct real123_x {real1_x a; real2_x b; real3_x c;} real123_x;

#ifdef DOUBLE_V
typedef double real_v;
typedef double3 real3_v;
#define MYMPI_REAL_v MPI_DOUBLE
#else
typedef real real_v;
typedef real3 real3_v;
#define MYMPI_REAL_V MYMPI_REAL
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
