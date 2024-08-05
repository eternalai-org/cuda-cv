#ifndef __UTILITIES_H__
#define __UTILITIES_H__

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

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

template <class T> 
std::ostream &operator << (std::ostream &s, const std::vector<T> &a) {
	s << "[";

    for (int i = 0; i < a.size() - 1; ++i)
    {
        s << a[i] << ", ";
    }

    if (a.size() > 0)
    {
        s << a[a.size() - 1];
    }

	return s << "]";
}

// std::ostream &operator << (std::ostream &s, const Tensor& a) {
//     s << "Tensor(" << a.mshape << ")";
//     return s;
// }


#endif // __UTILITIES_H__