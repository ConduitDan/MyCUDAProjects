CC=g++
NVCC=nvcc
CFLAGS=-I.
DEPS = Gradient.hpp Mesh.hpp ShapeOptimizer.hpp APIFactory.hpp CUDA_API.hpp OpenCL_API.hpp 
OBJ =  main.o ShapeOptimizer.o Gradient.o Mesh.o

%.o: %.cpp $(DEPS)
	$(CC)  -c $(CFLAGS) -o $@ $<  $(LDFLAGS)

%.o: %.cu $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(LDFLAGS)

openCL: LDFLAGS += -l OpenCL
openCL: CC = g++
openCL: CFLAGS += -D__OPENCL__
openCL: main.o ShapeOptimizer.o Gradient.o OpenCL_API.o Mesh.o APIFactory.o
	$(CC) $(CFLAGS) -o  optCL.out $^ $(LDFLAGS)
	make clean

checkopenCL: LDFLAGS += -l OpenCL
checkopenCL: CFLAGS += -D__OPENCL__
checkopenCL: GPU_API_Tests.o ShapeOptimizer.o Gradient.o OpenCL_API.o Mesh.o APIFactory.o
	$(CC) $(CFLAGS) -o test.out $^ $(LDFLAGS)
	make clean
	./test.out


cuda: CC = nvcc
cuda: main.o ShapeOptimizer.o Gradient.o CUDA_API.o Mesh.o APIFactory.o 
	$(CC) -o optCUDA.out $^ 
	make clean
	make checkcuda
checkcuda: CC= nvcc
checkcuda: GPU_API_Tests.o ShapeOptimizer.o Gradient.o CUDA_API.o Mesh.o APIFactory.o 
	$(CC) -o test.out $^
	./test.out
	make clean


.PHONY: clean
clean:
	rm -rf *.o


debugCL: CFLAGS += -DDEBUG -g
debugCL: checkopenCL


debug: CFLAGS += -DDEBUG -g
debug: cuda

