#ifndef __UTILITIES_H__
#define __UTILITIES_H__

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

void printmat3d(long long* mat, int h, int w, int c);

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#endif // __UTILITIES_H__