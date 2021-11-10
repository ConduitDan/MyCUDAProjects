#pragma once
#ifndef GPUShapeOptimizer_hpp
#define GPUShapeOptimizer_hpp

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

#include <Windowsnumerics.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>  
#include <string.h>
#include <vector>

#include "myMeshTools.hpp"



class GPUShapeOptimizer {


private:
    unsigned int meshSize = 0;
    unsigned int facetSize = 0;
 
    float3* vertices = NULL;
    unsigned int* facets = NULL;

    std::vector < std::tuple<unsigned int, unsigned int>>* vertex2facet = NULL;

    float3* dev_vertices = 0;
    unsigned int* dev_facets = 0;
    float* dev_areaPerFace = 0;
    float* dev_areaSum = 0;
    float* areaPerFace = NULL;
    bool readSuccess;

    cudaError_t cudaStatus;

public: 
    GPUShapeOptimizer(const char*);

    double calculateArea();
    void calculateGradArea();
    double calculateVolume();
    void calculateGradVolume();
    
    void printMesh();

 

    


    // getters
    unsigned int get_meshSize() { return meshSize; }
    unsigned int get_facetSize() { return facetSize; }
    float3* get_vertices() { return vertices; }
    unsigned int* get_facets() { return vertices; }
    bool get_readSuccess() { return readSuccess; }

};