cu_header = -I/usr/local/cuda/include
header = -Iinclude
header_files = include/fixedlonglong32x32.cuh include/functional.h
optimization = -O2
compile_flags = -Xcompiler="-fpermissive"
CPP_VERSION = -std=c++17

kernels.o: algo/kernels.cu
	nvcc --compiler-options -fPIC -c -o kernels.o  algo/kernels.cu \
	$(cu_header) $(header) $(optimization) $(compile_flags) $(CPP_VERSION)

matrixmul.o: algo/matrixmul.cu
	nvcc --compiler-options -fPIC -c -o matrixmul.o  algo/matrixmul.cu \
	$(cu_header) $(header) $(optimization) $(compile_flags) $(CPP_VERSION)

element_wise_operator.o:  algo/element_wise_operator.cu
	nvcc --compiler-options -fPIC -c -o element_wise_operator.o  algo/element_wise_operator.cu \
	$(cu_header) $(header) $(optimization) $(compile_flags) $(CPP_VERSION)

normalization.o:  algo/normalization.cu
	nvcc --compiler-options -fPIC -c -o normalization.o  algo/normalization.cu \
	$(cu_header) $(header) $(optimization) $(compile_flags) $(CPP_VERSION)

conv.o:  algo/conv.cu
	nvcc --compiler-options -fPIC -c -o conv.o  algo/conv.cu \
	$(cu_header) $(header) $(optimization) $(compile_flags) $(CPP_VERSION)

pooling.o:  algo/pooling.cu
	nvcc --compiler-options -fPIC -c -o pooling.o  algo/pooling.cu \
	$(cu_header) $(header) $(optimization) $(compile_flags) $(CPP_VERSION)

activations.o:  algo/activations.cu
	nvcc --compiler-options -fPIC -c -o activations.o  algo/activations.cu \
	$(cu_header) $(header) $(optimization) $(compile_flags) $(CPP_VERSION)

merging.o:  algo/merging.cu
	nvcc --compiler-options -fPIC -c -o merging.o  algo/merging.cu \
	$(cu_header) $(header) $(optimization) $(compile_flags) $(CPP_VERSION)

reduction.o:  algo/reduction.cu
	nvcc --compiler-options -fPIC -c -o reduction.o  algo/reduction.cu \
	$(cu_header) $(header) $(optimization) $(compile_flags) $(CPP_VERSION)

wrapper.o: deprecated/algo/wrapper.cu
	nvcc --compiler-options -fPIC -c -o wrapper.o deprecated/algo/wrapper.cu \
	$(cu_header) $(header) $(optimization) $(compile_flags) $(CPP_VERSION)

computelib.o: computelib.cu
	nvcc --compiler-options -fPIC -c -o computelib.o computelib.cu \
	$(cu_header) $(header) $(optimization) $(compile_flags) $(CPP_VERSION)

libcomputelib.so: matrixmul.o element_wise_operator.o normalization.o conv.o pooling.o \
	activations.o merging.o kernels.o reduction.o wrapper.o computelib.o
	nvcc --ptxas-options=-v -o libcomputelib.so --shared matrixmul.o element_wise_operator.o normalization.o \
	conv.o pooling.o activations.o merging.o kernels.o reduction.o wrapper.o computelib.o \
	$(cu_header) $(header) $(optimization) $(compile_flags) $(CPP_VERSION)

helpers.o: utils/helpers.cpp
	nvcc --compiler-options -fPIC -c -o helpers.o utils/helpers.cpp \
	$(cu_header) $(header) $(optimization) $(compile_flags) $(CPP_VERSION)

main: main.cu libcomputelib.so helpers.o
	nvcc -o main main.cu helpers.o -L. -lcomputelib 
	$(cu_header) $(header) $(optimization)

exec: main
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(pwd) ./main

debug: libcomputelib.so
	python3 ../test/test_abi.py

release: libcomputelib.so
	if [ -d "build" ]; then rm -rf build; fi
	mkdir build
	mkdir build/include
	mv libcomputelib.so build
	cp include/operations.cuh build/include
	cp include/computelib.h build/include

clean:
	rm -f *.o *.so