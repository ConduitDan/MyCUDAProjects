#pragma once
#ifndef Kernals_hpp
#define Kernals_hpp
#include "cuda.h"
#include "cuda_runtime.h"
#include <stdio.h> 

__global__ void areaKernel(double * area, double * vert, unsigned int * facets, unsigned int numFacets);
__global__ void volumeKernel(double * volume, double * vert, unsigned int * facets, unsigned int numFacets);
__global__ void addTree(double * in, double * out);
__global__ void addWithMultKernel(double *a ,double *b,double lambda, unsigned int size); // a += b * lambda
__global__ void areaGradient(double* gradAFacet, unsigned int* facets,double* verts,unsigned int numFacets);
__global__ void volumeGradient(double* gradVFacet, unsigned int* facets,double* verts,unsigned int numFacets);
__global__ void facetToVertex(double* vertexValue, double* facetValue,unsigned int* vertToFacet, unsigned int* vertIndexStart,unsigned int numVert);
__global__ void projectForce(double* force,double* gradAVert,double* gradVVert,double scale,unsigned int numEle);
__global__ void elementMultiply(double* v1, double* v2, double* out, unsigned int size);

__device__ void vectorSub(double * v1, double * v2, double * vOut);
__device__ void vectorAdd(double * v1, double * v2, double * vOut);
__device__ void vecScale(double *v, double lambda);
__device__ void vecAssign(double *out, double *in,double lambda); // out  = in*lambda
__device__ void cross(double *a,double *b, double *c);
__device__ double dot(double *a, double *b, double *c);
__device__ double norm(double *a); 
__device__ int sign(double a);

// template <unsigned int blockSize> __device__ void warpReduce(volatile double *sdata, unsigned int tid);
// template <unsigned int blockSize> __global__ void reduce6(double *g_idata,double *g_odata, unsigned int n);

void cuda_sync_and_check(cudaError_t cudaStatus, const char * caller);
double sum_of_elements(cudaError_t cudaStatus,double* vec,unsigned int size,unsigned int bufferedSize,unsigned int blockSize);
double dotProduct(cudaError_t cudaStatus,double * v1, double * v2, double * scratch, unsigned int size, unsigned int blockSize);


#endif