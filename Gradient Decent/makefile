CC=g++
NVCC=nvcc
CFLAGS=-I.
DEPS = Gradient.hpp Mesh.hpp ShapeOptimizer.hpp APIFactory.hpp CUDA_API.hpp OpenCL_API.hpp
OBJ =  main.o ShapeOptimizer.o Gradient.o Kernals.o Mesh.o

%.o: %.cpp $(DEPS)
	$(NVCC) -c -o $@ $< $(CFLAGS)

%.o: %.cu $(DEPS)
	$(NVCC) -c -o $@ $< $(CFLAGS)


openCL:
	CFLAGS += -l OpenCL
	$(CC) main.o ShapeOptimizer.o Gradient.o OpenCL_API.o Mesh.o APIFactory.o
	make clean

cuda: main.o ShapeOptimizer.o Gradient.o CUDA_API.o Mesh.o APIFactory.o
	$(NVCC) -o opt.out $^
	make clean

.PHONY: clean
clean:
	rm -rf *.o

check: CUDATests.o ShapeOptimizer.o Gradient.o CUDA_API.o Mesh.o APIFactory.o
	$(NVCC) -g -o test.out $^
	./test.out
	make clean

debug: CFLAGS += -DDEBUG -g
debug: cuda

