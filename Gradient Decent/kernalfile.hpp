#pragma once
#ifndef KERNAL_HPP
#define KERNAL_HPP

#include <math.h>


#ifdef __NVCC__
#include "cuda.h"
#include "cuda_runtime.h"

#define DEVTAG 
#define HEADERTAG
#define GETID blockDim.x * blockIdx.x + threadIdx.x
#define BLOCKID blockDim.x * blockIdx.x
#define WORKITEMID threadIdx.x
#define STRHEADERTAG //
#define DPREFACTOR __device__

#define PREAMBLE(NAME) __global__ 

#define PREP_FOR_PARSE(ARG) ARG
#define CUDASHARE extern __shared__ double sdata[];
#define OPENCLSHARED


 __global__ void areaKernel(double * area, double * vert, unsigned int * facets, unsigned int numFacets);
 __global__ void volumeKernel(double * volume, double * vert, unsigned int * facets, unsigned int numFacets);
 __global__ void addTree(double * in, double * out);
 __global__ void addWithMultKernel(double *a ,double *b,double lambda, unsigned int size); // a += b * lambda
 __global__ void areaGradient(double* gradAFacet, unsigned int* facets,double* verts,unsigned int numFacets);
 __global__ void volumeGradient(double* gradVFacet, unsigned int* facets,double* verts,unsigned int numFacets);
 __global__ void facetToVertex(double* vertexValue, double* facetValue,unsigned int* vertToFacet, unsigned int* vertIndexStart,unsigned int numVert);
 __global__ void projectForce(double* force,double* gradAVert,double* gradVVert,double scale,unsigned int numEle);
 __global__ void elementMultiply(double* v1, double* v2, double* out, unsigned int size);



#elif 1
//defined __OPENCL_VERSION__
#define DEVTAG __global
#define HEADERTAG //
#define STRHEADERTAG
#define GETID get_global_id()

#define PREAMBLE(NAME) const char* Kernals::NAME ## STR = 

#define STR(ARG) #ARG;

#define PREP_FOR_PARSE(ARG) STR(ARG)

#define CLASS(ARG) Kernals::ARG

#define DPREFACTOR 

#define CUDASHARE 
#define OPENCLSHARED , __local double * sdata



#include "CL/cl.h"





class Kernals{

public:

DPREFACTOR void vectorSub(double * v1, double * v2, double * vOut);
DPREFACTOR void vectorAdd(double * v1, double * v2, double * vOut);
DPREFACTOR void vecScale(double *v, double lambda);
DPREFACTOR void vecAssign(double *out, double *in,double lambda);
DPREFACTOR void cross(double *a,double *b, double *c) ;
DPREFACTOR double dot(double *a, double *b) ;

DPREFACTOR double norm(double *a);

DPREFACTOR int sign(double a);

static const char* 	areaKernelSTR;
static const char* 	volumeKernelSTR;
static const char* 	addTreeSTR;
static const char* 	addWithMultKernelSTR;
static const char* 	areaGradientSTR;
static const char* 	volumeGradientSTR;
static const char* 	facetToVertexSTR;
static const char* 	projectForceSTR;
static const char* 	elementMultiplySTR;
};

#else
#error A CUDA or OpenCL compiler is required!

#endif





DPREFACTOR void vectorSub(double * v1, double * v2, double * vOut);
DPREFACTOR void vectorAdd(double * v1, double * v2, double * vOut);
DPREFACTOR void vecScale(double *v, double lambda);
DPREFACTOR void vecAssign(double *out, double *in,double lambda); // out  = in*lambda
DPREFACTOR void cross(double *a,double *b, double *c);
DPREFACTOR double dot(double *a, double *b, double *c);
DPREFACTOR double norm(double *a); 
DPREFACTOR int sign(double a);

#endif