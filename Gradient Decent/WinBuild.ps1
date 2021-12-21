nvcc -o opt.exe main.cpp ShapeOptimizer.cpp Gradient.cu Kernals.cu Mesh.cu
nvcc -o test.exe Tests.cu ShapeOptimizer.cpp Gradient.cu Kernals.cu Mesh.cu
./test.exe