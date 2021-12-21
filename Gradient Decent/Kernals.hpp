#pragma once
#ifndef Kernals_hpp
#define Kernals_hpp
#include "cuda.h"
#include "cuda_runtime.h"
#include <stdio.h> 

typedef void (*fptr)(void * devicePointer);

class DeviceAPI{
private: 
    static DeviceAPI* _instance; // this holds the pointer to the only instance of this class

protected:
    DeviceAPI();


public:
    static DeviceAPI* Instance(); 

    virtual void* allocate(unsigned int size) = 0;
    virtual void copy_to_host(void * hostPointer, void * devicepointer, unsigned int size) = 0;
    virtual void copy_to_device(void* devicePointer, void* hostPointer, unsigned int size) = 0;
    //virtual void deallocate(void* devicePointer) = 0;
    virtual fptr get_deallocate() = 0;


    virtual double sum_of_elements(double* vec,unsigned int size,unsigned int bufferedSize,unsigned int blockSize) = 0;
    virtual double dotProduct(double * v1, double * v2, double * scratch, unsigned int size, unsigned int blockSize) = 0;
    virtual void add_with_mult(double * a,double * b, double lambda, unsigned int size) = 0;//a = a + b* lambda
    virtual void project_force(double* force,double *gradAVert,double * gradVVert, double scale,unsigned int size) = 0;
    virtual void facet_to_vertex(double* vertexValue, double* facetValue,unsigned int* vertToFacet, unsigned int* vertIndexStart,unsigned int numVert) = 0;
    virtual void area_gradient(double * gradAFacet,unsigned int* facets,double * vert,unsigned int numFacets) = 0;
    virtual void volume_gradient(double * gradVFacet,unsigned int* facets,double * vert,unsigned int numFacets) = 0;

};


class CUDA: public DeviceAPI{

private:
    static cudaError_t cudaStatus;


public:
    void* allocate(unsigned int size);
    void copy_to_host(void * hostPointer, void * devicepointer, unsigned int size);
    void copy_to_device(void* devicePointer, void* hostPointer, unsigned int size);
    static void deallocate(void* devicePointer);
    fptr get_deallocate(){return &deallocate;}

    // template <unsigned int blockSize> __device__ void warpReduce(volatile double *sdata, unsigned int tid);
    // template <unsigned int blockSize> __global__ void reduce6(double *g_idata,double *g_odata, unsigned int n);

    void cuda_sync_and_check(const char * caller);
    double sum_of_elements(double* vec,unsigned int size,unsigned int bufferedSize,unsigned int blockSize);
    double dotProduct(double * v1, double * v2, double * scratch, unsigned int size, unsigned int blockSize);
    void add_with_mult(double * a,double * b, double lambda, unsigned int size);//a = a + b* lambda
    
    void project_force(double* force,double *gradAVert,double * gradVVert, double scale,unsigned int size);
    void facet_to_vertex(double* vertexValue, double* facetValue,unsigned int* vertToFacet, unsigned int* vertIndexStart,unsigned int numVert);

    void area_gradient(double * gradAFacet,unsigned int* facets,double * vert,unsigned int numFacets);
    void volume_gradient(double * gradVFacet,unsigned int* facets,double * vert,unsigned int numFacets);

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

    __device__ void vectorSub(double * v1, double * v2, double * vOut);
    __device__ void vectorAdd(double * v1, double * v2, double * vOut);
    __device__ void vecScale(double *v, double lambda);
    __device__ void vecAssign(double *out, double *in,double lambda); // out  = in*lambda
    __device__ void cross(double *a,double *b, double *c);
    __device__ double dot(double *a, double *b, double *c);
    __device__ double norm(double *a); 
    __device__ int sign(double a);

#endif