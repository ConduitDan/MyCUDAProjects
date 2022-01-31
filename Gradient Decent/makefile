CC=g++
NVCC=nvcc
CFLAGS=-I. 
DEPS = Gradient.hpp Mesh.hpp ShapeOptimizer.hpp APIFactory.hpp CUDA_API.hpp OpenCL_API.hpp cuBLASAPI.hpp
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
	$(CC) $(CFLAGS) -o optCUDA.out $^ 
	make clean
	
cudaATOMIC: CC = nvcc
cudaATOMIC: LDFLAGS += -D__ATOMIC__
cudaATOMIC: main.o ShapeOptimizer.o Gradient.o CUDA_API.o Mesh.o APIFactory.o 
	$(CC) $(CFLAGS) -o optCUDAAtomic.out $^ $(LDFLAGS)
	make clean

cudaATOMIC2: CC = nvcc
cudaATOMIC2: LDFLAGS += -D__ATOMIC__
cudaATOMIC2: main.o ShapeOptimizer.o Gradient.o CUDA_API.o Mesh.o APIFactory.o 
	$(CC) $(CFLAGS) -o optCUDAAtomic2.out $^ $(LDFLAGS)
	make clean


cuBLAS: CC = nvcc
cuBLAS: CFLAGS += -lcusparse -lcublas
cuBLAS: LDFLAGS += -lcusparse -lcublas -D__cuBLASopt__
cuBLAS: main.o ShapeOptimizer.o Gradient.o CUDA_API.o cuBLASAPI.o Mesh.o APIFactory.o 
	$(CC) $(CFLAGS) -o optcuBLAS.out $^ $(LDFLAGS)
	make clean


cuBLASATOMIC: CC = nvcc
cuBLASATOMIC: CFLAGS += -lcusparse -lcublas
cuBLASATOMIC: LDFLAGS += -lcusparse -lcublas -D__cuBLASopt__ -D__ATOMIC__
cuBLASATOMIC: main.o ShapeOptimizer.o Gradient.o CUDA_API.o cuBLASAPI.o Mesh.o APIFactory.o 
	$(CC) $(CFLAGS) -o optcuBLASAtomic2.out $^ $(LDFLAGS)
	make clean

checkcuBLAS: CC = nvcc
checkcuBLAS: CFLAGS += -lcusparse -lcublas
checkcuBLAS: LDFLAGS += -lcusparse -lcublas -D__cuBLASopt__
checkcuBLAS: GPU_API_Tests.o ShapeOptimizer.o Gradient.o CUDA_API.o Mesh.o APIFactory.o cuBLASAPI.o
	$(CC) $(CFLAGS) -o test.out $^ $(LDFLAGS)
	make clean
	./test.out

checkcuda: CC= nvcc
checkcuda: GPU_API_Tests.o ShapeOptimizer.o Gradient.o CUDA_API.o Mesh.o APIFactory.o
	$(CC) $(CFLAGS) -arch=sm_60 -o test.out $^ $(LDFLAGS)
	make clean
	./test.out



.PHONY: clean
clean:
	rm -rf *.o


debugCL: CFLAGS += -DDEBUG -g
debugCL: checkopenCL


debug: CFLAGS += -DDEBUG -g
debug: cuda
debug: checkcuda


debugcuBLAS: CFLAGS += -DDEBUG -g
debugcuBLAS: checkcuBLAS


