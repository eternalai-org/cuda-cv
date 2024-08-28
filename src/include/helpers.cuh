#ifndef __UTILITIES_H__
#define __UTILITIES_H__

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <tensor.h>
#include <vector>
#include <iostream>
#include <fixedlonglong32x32.cuh>

template<class t>
class array_view {
public:
    t *begin, *end;
    array_view(t *begin, t *end) : begin(begin), end(end) {}
    array_view(t *begin, int size) : begin(begin), end(begin + size) {}
    array_view(std::vector<t> &v) : begin(v.data()), end(v.data() + v.size()) {}
};

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

std::ostream &operator << (std::ostream &s, const FixedLongLong::FixedLongLongType &a) {
	return s << 1.0f * a.value / FixedLongLong::ONE;
}

template <class T> 
std::ostream &operator << (std::ostream &s, const std::vector<T> &a) {
	s << "[";

    for (const auto& x: a)
    {
        s << x << " ";
    }

	return s << "]";
}

template <class T> 
std::ostream &operator << (std::ostream &s, const array_view<T>& a) {
	for (T* it = a.begin; it != a.end; ++it)
    {
        s << *it;

        if (it != a.end - 1)
        {
            s << ", ";
        }
    }
    
    return s;
}

// std::ostream &operator << (std::ostream &s, const TensorWrapper& a) {
//     s << "Tensor(" << a.shape() << ") : ";
//     int prod = 1;
//     for (const auto& x: a.shape())
//     {
//         prod *= x;
//     }

//     const int64_t* ref = a.data();

//     for (int i = 0; i < prod; ++i)
//     {
//         s << ref[i] << " ";
//     }

//     return s;
// }


#endif // __UTILITIES_H__