#!/bin/bash
nvcc -o opt.out main.cpp ShapeOptimizer.cpp Gradient.cu Kernals.cu Mesh.cu
