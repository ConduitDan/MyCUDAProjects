CC=g++
NVCC=nvcc
CFLAGS=-I.
DEPS = Gradient.hpp Mesh.hpp ShapeOptimizer.hpp Kernals.hpp
OBJ =  main.o ShapeOptimizer.o Gradient.o Kernals.o Mesh.o

%.o: %.cpp $(DEPS)
	$(NVCC) -c -o $@ $< $(CFLAGS)

%.o: %.cu $(DEPS)
	$(NVCC) -c -o $@ $< $(CFLAGS)

cudamake: main.o ShapeOptimizer.o Gradient.o Kernals.o Mesh.o
	$(NVCC) -o opt.out $^
	make clean

.PHONY: clean
clean:
	rm -rf *.o

check: Tests.o ShapeOptimizer.o Gradient.o Kernals.o Mesh.o
	$(NVCC) -o test.out $^
	./test.out
	make clean

debug: CFLAGS += -DDEBUG -g
debug: cudamake

