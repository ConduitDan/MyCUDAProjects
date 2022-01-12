CC=g++
NVCC=nvcc
CFLAGS=-I.
DEPS = Gradient.hpp Mesh.hpp ShapeOptimizer.hpp APIFactory.hpp CUDA_API.hpp OpenCL_API.hpp kernalfile.hpp
OBJ =  main.o ShapeOptimizer.o Gradient.o Kernals.o Mesh.o

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(LDFLAGS)

%.o: %.cu $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

openCL: LDFLAGS += -l OpenCL
openCL: main.o ShapeOptimizer.o Gradient.o OpenCL_API.o Mesh.o APIFactory.o kernalfile.o
	CC = g++
	$(CC) -o opt.out $^ $(LDFLAGS)
	make clean


.PHONY: clean
clean:
	rm -rf *.o

cuda: main.o ShapeOptimizer.o Gradient.o CUDA_API.o Mesh.o APIFactory.o kernalfile.o
	CC = nvcc
	$(CC) -o opt.out $^
	make clean

check: CUDATests.o ShapeOptimizer.o Gradient.o CUDA_API.o Mesh.o APIFactory.o kernalfile.o
	$(NVCC) -g -o test.out $^
	./test.out
	make clean

debug: CFLAGS += -DDEBUG -g
debug: cuda

