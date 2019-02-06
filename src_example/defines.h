#ifndef MD_RNA_IONS_DEFINES_H
#define MD_RNA_IONS_DEFINES_H

#define MAXLENGTH 4096
// 1 Molar in nm^-3
#define MOLAR 0.6022
#define DIM3 3
#define M_SQRT8PI3 sqrt(8*M_PI*M_PI*M_PI)

// #define DOUBLEFLAG

#ifdef DOUBLEFLAG
typedef double real;
typedef double2 real2;
typedef double3 real3;
typedef double4 real4;
#define make_real3 make_double3
#define realRecip __drcp_rn
#define sinreal sin
#define cosreal cos
#define sqrtreal sqrt
typedef long int fixreal; // usually unsigned
typedef unsigned long int unsignedfixreal; // usually unsigned
#define FREAL_MAX (-2.0*((real) LONG_MIN))
typedef ulong3 ufreal3;
#else
typedef float real;
typedef float2 real2;
typedef float3 real3;
typedef float4 real4;
#define make_real3 make_float3
#define realRecip __frcp_rn
// One of these three causes energy drift over 20k steps in the dihedrals
// #define sinreal __sinf
// #define cosreal __cosf
// #define sqrtreal __fsqrt_rn
#define sinreal sin
#define cosreal cos
#define sqrtreal sqrt
typedef int fixreal; // usually unsigned
typedef unsigned int unsignedfixreal; // usually unsigned
#define FREAL_MAX (-2.0*((real) INT_MIN))
typedef uint3 ufreal3;
#endif

// Update block size
#define BU 256
// Bonded block size
#define BB 256
// Nonbonded block size
// #define TNB 32
#define TLJ 32
#define TES 8
#define BNB 256

#define XHASH 32

#if (BNB>BU) || (BB>BU)
#error "Error in defines.h. BU must be greater than or equal to BNB and BB"
#endif

#endif

