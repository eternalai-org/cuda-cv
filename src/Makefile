cu_header = -I/usr/local/cuda/include
header = -Iinclude
header_files = include/fixedlonglong32x32.cuh include/functional.h
optimization = -O2
compile_flags = -Xcompiler="-fpermissive"
CPP_VERSION = -std=c++17
CPP_SYMBOLS = -DLOGGING_DEBUG

kernels.o: algo/kernels.cu
	nvcc --compiler-options -fPIC -c -o kernels.o  algo/kernels.cu \
	$(cu_header) $(header) $(optimization) $(compile_flags) $(CPP_VERSION) $(CPP_SYMBOLS)

matrixmul.o: algo/matrixmul.cu
	nvcc --compiler-options -fPIC -c -o matrixmul.o  algo/matrixmul.cu \
	$(cu_header) $(header) $(optimization) $(compile_flags) $(CPP_VERSION) $(CPP_SYMBOLS)

element_wise_operator.o:  algo/element_wise_operator.cu
	nvcc --compiler-options -fPIC -c -o element_wise_operator.o  algo/element_wise_operator.cu \
	$(cu_header) $(header) $(optimization) $(compile_flags) $(CPP_VERSION) $(CPP_SYMBOLS)

normalization.o:  algo/normalization.cu
	nvcc --compiler-options -fPIC -c -o normalization.o  algo/normalization.cu \
	$(cu_header) $(header) $(optimization) $(compile_flags) $(CPP_VERSION) $(CPP_SYMBOLS)

conv.o:  algo/conv.cu
	nvcc --compiler-options -fPIC -c -o conv.o  algo/conv.cu \
	$(cu_header) $(header) $(optimization) $(compile_flags) $(CPP_VERSION) $(CPP_SYMBOLS)

pooling.o:  algo/pooling.cu
	nvcc --compiler-options -fPIC -c -o pooling.o  algo/pooling.cu \
	$(cu_header) $(header) $(optimization) $(compile_flags) $(CPP_VERSION) $(CPP_SYMBOLS)

activations.o:  algo/activations.cu
	nvcc --compiler-options -fPIC -c -o activations.o  algo/activations.cu \
	$(cu_header) $(header) $(optimization) $(compile_flags) $(CPP_VERSION) $(CPP_SYMBOLS)

merging.o:  algo/merging.cu
	nvcc --compiler-options -fPIC -c -o merging.o  algo/merging.cu \
	$(cu_header) $(header) $(optimization) $(compile_flags) $(CPP_VERSION) $(CPP_SYMBOLS)

reduction.o:  algo/reduction.cu
	nvcc --compiler-options -fPIC -c -o reduction.o  algo/reduction.cu \
	$(cu_header) $(header) $(optimization) $(compile_flags) $(CPP_VERSION) $(CPP_SYMBOLS)

wrapper.o: deprecated/algo/wrapper.cu
	nvcc --compiler-options -fPIC -c -o wrapper.o deprecated/algo/wrapper.cu \
	$(cu_header) $(header) $(optimization) $(compile_flags) $(CPP_VERSION) $(CPP_SYMBOLS)

computelib.o: computelib.cu
	nvcc --compiler-options -fPIC -c -o computelib.o computelib.cu \
	$(cu_header) $(header) $(optimization) $(compile_flags) $(CPP_VERSION) $(CPP_SYMBOLS)

libcomputelib.so: matrixmul.o element_wise_operator.o normalization.o conv.o pooling.o \
	activations.o merging.o kernels.o reduction.o wrapper.o computelib.o
	nvcc --ptxas-options=-v -o libcomputelib.so --shared matrixmul.o element_wise_operator.o normalization.o \
	conv.o pooling.o activations.o merging.o kernels.o reduction.o wrapper.o computelib.o \
	$(cu_header) $(header) $(optimization) $(compile_flags) $(CPP_VERSION) $(CPP_SYMBOLS)

helpers.o: utils/helpers.cpp
	nvcc --compiler-options -fPIC -c -o helpers.o utils/helpers.cpp \
	$(cu_header) $(header) $(optimization) $(compile_flags) $(CPP_VERSION) $(CPP_SYMBOLS)

main: main.cu libcomputelib.so helpers.o
	nvcc -o main main.cu helpers.o -L. -lcomputelib 
	$(cu_header) $(header) $(optimization) $(CPP_SYMBOLS)

exec: main
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(pwd) ./main

test_abi: libcomputelib.so	
	python3 -X nogil ../test/test_abi.py

test_act: libcomputelib.so
	python3 -X nogil ../test/test_activations.py

test_elementwise: libcomputelib.so
	python3 -X nogil ../test/test_elementwise.py

test_matmul: libcomputelib.so
	python3 -X nogil ../test/test_matmul.py

test_transformation: libcomputelib.so
	python3 -X nogil ../test/test_transformation.py

test_merging: libcomputelib.so
	TF_CPP_MIN_LOG_LEVEL=3 python3 -X nogil ../test/test_merging.py

test_pooling: libcomputelib.so
	TF_CPP_MIN_LOG_LEVEL=3 python3 -X nogil ../test/test_pooling.py

test_conv2d: libcomputelib.so
	TF_CPP_MIN_LOG_LEVEL=3 python3 -X nogil ../test/test_conv2d.py

test_normalization: libcomputelib.so
	TF_CPP_MIN_LOG_LEVEL=3 python3 -X nogil ../test/test_normalization.py

release: libcomputelib.so
	if [ -d "build" ]; then rm -rf build; fi
	mkdir build
	mkdir build/include
	mv libcomputelib.so build
	cp include/operations.cuh build/include
	cp include/computelib.h build/include

clean:
	rm -f *.o *.so