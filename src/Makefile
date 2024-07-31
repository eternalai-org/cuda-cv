cu_header = -I/usr/local/cuda/include
header = -Iinclude
header_files = include/fixedlonglong32x32.cuh include/functional.h
optimization = -O2

kernels.o: algo/kernels.cu
	nvcc --compiler-options -fPIC -c -o kernels.o  algo/kernels.cu $(cu_header) $(header) $(optimization)

matrixmul.o: algo/matrixmul.cu
	nvcc --compiler-options -fPIC -c -o matrixmul.o  algo/matrixmul.cu $(cu_header) $(header) $(optimization)

element_wise_operator.o:  algo/element_wise_operator.cu
	nvcc --compiler-options -fPIC -c -o element_wise_operator.o  algo/element_wise_operator.cu $(cu_header) $(header) $(optimization)

normalization.o:  algo/normalization.cu
	nvcc --compiler-options -fPIC -c -o normalization.o  algo/normalization.cu $(cu_header) $(header) $(optimization)

conv.o:  algo/conv.cu
	nvcc --compiler-options -fPIC -c -o conv.o  algo/conv.cu $(cu_header) $(header) $(optimization)

pooling.o:  algo/pooling.cu
	nvcc --compiler-options -fPIC -c -o pooling.o  algo/pooling.cu $(cu_header) $(header) $(optimization)

activations.o:  algo/activations.cu
	nvcc --compiler-options -fPIC -c -o activations.o  algo/activations.cu $(cu_header) $(header) $(optimization)

merging.o:  algo/merging.cu
	nvcc --compiler-options -fPIC -c -o merging.o  algo/merging.cu $(cu_header) $(header) $(optimization)

reduction.o:  algo/reduction.cu
	nvcc --compiler-options -fPIC -c -o reduction.o  algo/reduction.cu $(cu_header) $(header) $(optimization)

libcomputelib.so: matrixmul.o element_wise_operator.o normalization.o conv.o pooling.o activations.o merging.o kernels.o reduction.o
	nvcc --ptxas-options=-v -o libcomputelib.so --shared matrixmul.o element_wise_operator.o normalization.o \
			conv.o pooling.o activations.o merging.o kernels.o reduction.o $(cu_header) $(header) $(optimization)

utilities.o: utilities.cpp
	nvcc --compiler-options -fPIC -c -o utilities.o utilities.cpp $(cu_header) $(header) $(optimization)

main: main.cu libcomputelib.so utilities.o
	nvcc -o main main.cu utilities.o -L. -lcomputelib $(cu_header) $(header) $(optimization)

exec: main
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(pwd) ./main

clean:
	rm -f *.o *.so