#!/bin/bash
nvcc -o opt.out main.cpp ShapeOptimizer.cpp Gradient.cu Kernals.cu Mesh.cu
nvcc -o test.out Tests.cu ShapeOptimizer.cpp Gradient.cu Kernals.cu Mesh.cu
./test.out