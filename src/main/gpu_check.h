#ifndef MAIN_GPU_CHECK_H
#define MAIN_GPU_CHECK_H

#include <stdio.h>

#define gpuCheck(cudaCommand) do {                                 \
  cudaError_t err = (cudaCommand);                                 \
  if (err != cudaSuccess) {                                        \
    fprintf(stderr, "Error in file %s, line %d\n",__FILE__,__LINE__);\
    fprintf(stderr, "Error string: %s\n",cudaGetErrorString(err)); \
    exit(err);                                                     \
  }                                                                \
} while(0)

/*#include <iostream>  // for gpuCheck

#define gpuCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        std::cerr << "GPUassert: " << cudaGetErrorString(code)
                  << " " << file
                  << " " << line
                  << std::endl;

        if (abort) exit(code);
    }
}*/

#endif
