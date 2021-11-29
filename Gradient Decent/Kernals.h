#pragma once
#ifndef Kernals_h
#define Kernals_h
#include "cuda.h"
#include "cuda_runtime.h"

__global__ void areaKernel(double * area, double * vert, unsigned int * facets, unsigned int numFacets);
__global__ void volumeKernel(double * volume, double * vert, unsigned int * facets, unsigned int numFacets);
__global__ void addTree(const double * in, double * out,unsigned int size);
__global__ void addWithMultKernel(double *a ,double *b,double lambda, unsigned int size); // a += b * lambda
__global__ void areaGradient(double* gradAFacet, unsigned int* facets,double* verts,unsigned int numFacets);
__global__ void volumeGradient(double* gradVFacet, unsigned int* facets,double* verts,unsigned int numFacets);
__global__ void facetToVertex(double* vertexValue, double* facetValue,unsigned int* vertToFacet, unsigned int* vertIndexStart,unsigned int numVert);
__global__ void projectForce(double* force,double* gradAVert,double* gradVVert,unsigned int numVert);


__device__ void vectorSub(double * v1, double * v2, double * vOut);
__device__ void vectorAdd(double * v1, double * v2, double * vOut);
__device__ void vecScale(double *v, double lambda);
__device__ void vecAssign(double *out, double *in,double lambda); // out  = in*lambda
__device__ void cross(double *a,double *b, double *c);
__device__ double dot(double *a, double *b, double *c);
__device__ double norm(double *a);


#endif