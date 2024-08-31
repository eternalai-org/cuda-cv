cu_headers = # -I/usr/local/cuda/include
HEADERS = -Iinclude $(cu_headers)
OPTIMIZATION = -O2
COMPILE_FLAGS = -Xcompiler="-fpermissive"
CPP_VERSION = -std=c++17
_CPP_SYMBOLS = -DLOGGING_DEBUG -DLOGGING_VERBOSE

ifeq ($(debug), 1)
	CPP_SYMBOLS=-DLOGGING_DEBUG -DLOGGING_VERBOSE
	
else
	CPP_SYMBOLS=
endif 

kernels.o: algo/kernels.cu
	nvcc --compiler-options -fPIC -c -o kernels.o  algo/kernels.cu \
	$(HEADERS) $(OPTIMIZATION) $(COMPILE_FLAGS) $(CPP_VERSION) $(CPP_SYMBOLS)

matrixmul.o: algo/matrixmul.cu
	nvcc --compiler-options -fPIC -c -o matrixmul.o  algo/matrixmul.cu \
	$(HEADERS) $(OPTIMIZATION) $(COMPILE_FLAGS) $(CPP_VERSION) $(CPP_SYMBOLS)

element_wise_operator.o:  algo/element_wise_operator.cu
	nvcc --compiler-options -fPIC -c -o element_wise_operator.o  algo/element_wise_operator.cu \
	$(HEADERS) $(OPTIMIZATION) $(COMPILE_FLAGS) $(CPP_VERSION) $(CPP_SYMBOLS)

normalization.o:  algo/normalization.cu
	nvcc --compiler-options -fPIC -c -o normalization.o  algo/normalization.cu \
	$(HEADERS) $(OPTIMIZATION) $(COMPILE_FLAGS) $(CPP_VERSION) $(CPP_SYMBOLS)

conv.o:  algo/conv.cu
	nvcc --compiler-options -fPIC -c -o conv.o  algo/conv.cu \
	$(HEADERS) $(OPTIMIZATION) $(COMPILE_FLAGS) $(CPP_VERSION) $(CPP_SYMBOLS)

pooling.o:  algo/pooling.cu
	nvcc --compiler-options -fPIC -c -o pooling.o  algo/pooling.cu \
	$(HEADERS) $(OPTIMIZATION) $(COMPILE_FLAGS) $(CPP_VERSION) $(CPP_SYMBOLS)

activations.o:  algo/activations.cu
	nvcc --compiler-options -fPIC -c -o activations.o  algo/activations.cu \
	$(HEADERS) $(OPTIMIZATION) $(COMPILE_FLAGS) $(CPP_VERSION) $(CPP_SYMBOLS)

merging.o:  algo/merging.cu
	nvcc --compiler-options -fPIC -c -o merging.o  algo/merging.cu \
	$(HEADERS) $(OPTIMIZATION) $(COMPILE_FLAGS) $(CPP_VERSION) $(CPP_SYMBOLS)

reduction.o:  algo/reduction.cu
	nvcc --compiler-options -fPIC -c -o reduction.o  algo/reduction.cu \
	$(HEADERS) $(OPTIMIZATION) $(COMPILE_FLAGS) $(CPP_VERSION) $(CPP_SYMBOLS)

wrapper.o: deprecated/algo/wrapper.cu
	nvcc --compiler-options -fPIC -c -o wrapper.o deprecated/algo/wrapper.cu \
	$(HEADERS) $(OPTIMIZATION) $(COMPILE_FLAGS) $(CPP_VERSION) $(CPP_SYMBOLS)

computelib.o: computelib.cu
	nvcc --compiler-options -fPIC -c -o computelib.o computelib.cu \
	$(HEADERS) $(OPTIMIZATION) $(COMPILE_FLAGS) $(CPP_VERSION) $(CPP_SYMBOLS)

helpers.o: utils/helpers.cu
	nvcc --compiler-options -fPIC -c -o helpers.o utils/helpers.cu \
	$(HEADERS) $(OPTIMIZATION) $(COMPILE_FLAGS) $(CPP_VERSION) $(CPP_SYMBOLS)

libcomputelib.so: matrixmul.o element_wise_operator.o normalization.o conv.o pooling.o \
	activations.o merging.o kernels.o reduction.o wrapper.o computelib.o helpers.o
	nvcc --ptxas-options=-v -o libcomputelib.so --shared matrixmul.o element_wise_operator.o normalization.o \
	conv.o pooling.o activations.o merging.o kernels.o reduction.o wrapper.o computelib.o helpers.o \
	$(HEADERS) $(OPTIMIZATION) $(COMPILE_FLAGS) $(CPP_VERSION) $(CPP_SYMBOLS)


main: main.cu libcomputelib.so helpers.o
	nvcc -o main main.cu helpers.o -L. -lcomputelib 
	$(HEADERS) $(OPTIMIZATION) $(CPP_SYMBOLS)

exec: main
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(pwd) ./main

test_abi: libcomputelib.so	
	python3 ../test/test_abi.py

test_act: libcomputelib.so
	python3 ../test/test_activations.py

test_elementwise: libcomputelib.so
	python3 ../test/test_elementwise.py

test_matmul: libcomputelib.so
	python3 ../test/test_matmul.py

test_transformation: libcomputelib.so
	python3 ../test/test_transformation.py

test_merging: libcomputelib.so
	TF_CPP_MIN_LOG_LEVEL=3 python3 ../test/test_merging.py

test_pooling: libcomputelib.so
	TF_CPP_MIN_LOG_LEVEL=3 python3 ../test/test_pooling.py

test_conv2d: libcomputelib.so
	TF_CPP_MIN_LOG_LEVEL=3 python3 ../test/test_conv2d.py

test_norm: libcomputelib.so
	TF_CPP_MIN_LOG_LEVEL=3 python3 ../test/test_norm.py

test_reduction: libcomputelib.so
	TF_CPP_MIN_LOG_LEVEL=3 python3 ../test/test_reduction.py

benchmark: clean libcomputelib.so
	TF_CPP_MIN_LOG_LEVEL=3 python3 ../test/benchmark.py

test_all: clean test_pooling test_conv2d test_norm test_reduction \
	test_merging test_transformation test_matmul test_elementwise \
	test_act test_abi

	echo "All tests passed"

release: clean libcomputelib.so
	if [ -d "build" ]; then rm -rf build; fi
	mkdir build
	mkdir build/include
	mv libcomputelib.so build
	cp include/operations.cuh build/include
	cp include/computelib.h build/include

git_release: release
	rm -rf ../release
	mv build ../release
	git add ../release
	git commit -m "[automatic] Release $(shell date)"
	git push

clean:
	rm -f *.o *.so
