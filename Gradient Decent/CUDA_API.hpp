#pragma once
#ifndef CUDA_API_hpp
#define CUDA_API_hpp



#include "DeviceAPI.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
#include <stdio.h> 


class CUDA: public DeviceAPI{

private:
    cudaError_t _cudaStatus;


public:
    CUDA();
    CUDA(int blocksize);
    //void* allocate(unsigned int size);
	void allocate(void** ptr, unsigned int size);
    void copy_to_host(void * hostPointer, void * devicepointer, unsigned int size);
    void copy_to_device(void* devicePointer, void* hostPointer, unsigned int size);
    void deallocate(void* devicePointer);
	double getGPUElement(double * vec, unsigned int index);
	unsigned int getGPUElement(unsigned int * vec, unsigned int index);

    // template <unsigned int blockSize> __device__ void warpReduce(volatile double *sdata, unsigned int tid);
    // template <unsigned int blockSize> __global__ void reduce6(double *g_idata,double *g_odata, unsigned int n);

    void cuda_sync_and_check(const char * caller);

	double sum_of_elements(UniqueDevicePtr<double>* vec,unsigned int size,unsigned int bufferedSize);
    double dotProduct(UniqueDevicePtr<double>* v1, UniqueDevicePtr<double>* v2, UniqueDevicePtr<double>* scratch, unsigned int size);
    void add_with_mult(UniqueDevicePtr<double>* a,UniqueDevicePtr<double>* b, double lambda, unsigned int size);//a = a + b* lambda
    
	void project_force(UniqueDevicePtr<double>* force,UniqueDevicePtr<double>* gradAVert,UniqueDevicePtr<double>* gradVVert, double scale,unsigned int size);
    void facet_to_vertex(UniqueDevicePtr<double>* vertexValue, UniqueDevicePtr<double>* facetValue,UniqueDevicePtr<unsigned int>* vertToFacet, UniqueDevicePtr<unsigned int>* vertIndexStart,unsigned int numVert);
    
	void area_gradient(UniqueDevicePtr<double>* gradAFacet,UniqueDevicePtr<unsigned int>* facets,UniqueDevicePtr<double>* vert,unsigned int numFacets);
    void volume_gradient(UniqueDevicePtr<double>* gradVFacet,UniqueDevicePtr<unsigned int>* facets,UniqueDevicePtr<double>* vert,unsigned int numFacets);
    
	void area(UniqueDevicePtr<double>* area, UniqueDevicePtr<double>* vert, UniqueDevicePtr<unsigned int>* facets, unsigned int numFacets);
    void volume(UniqueDevicePtr<double>* volume, UniqueDevicePtr<double>* vert, UniqueDevicePtr<unsigned int>* facets, unsigned int numFacets);

	void area_gradientVert(UniqueDevicePtr<double>* gradAVert,UniqueDevicePtr<unsigned int>* facets,UniqueDevicePtr<double>* vert,unsigned int numFacets,unsigned int numVert);


};





    __global__ void areaKernel(double * area, double * vert, unsigned int * facets, unsigned int numFacets);
    __global__ void volumeKernel(double * volume, double * vert, unsigned int * facets, unsigned int numFacets);
    __global__ void addTree(double * in, double * out);
    __global__ void addWithMultKernel(double *a ,double *b,double lambda, unsigned int size); // a += b * lambda
    __global__ void areaGradient(double* gradAFacet, unsigned int* facets,double* verts,unsigned int numFacets);
    __global__ void volumeGradient(double* gradVFacet, unsigned int* facets,double* verts,unsigned int numFacets);
    __global__ void facetToVertex(double* vertexValue, double* facetValue,unsigned int* vertToFacet, unsigned int* vertIndexStart,unsigned int numVert);
    __global__ void projectForce(double* force,double* gradAVert,double* gradVVert,double scale,unsigned int numEle);
    __global__ void elementMultiply(double* v1, double* v2, double* out, unsigned int size);



    // __device__ void vectorSub(double * v1, double * v2, double * vOut);
    // __device__ void vectorAdd(double * v1, double * v2, double * vOut);
    // __device__ void vecScale(double *v, double lambda);
    // __device__ void vecAssign(double *out, double *in,double lambda); // out  = in*lambda
    // __device__ void cross(double *a,double *b, double *c);
    // __device__ double dot(double *a, double *b, double *c);
    // __device__ double norm(double *a); 
    // __device__ int sign(double a);


    

#endif