#pragma once
#ifndef OpenCL_API_hpp
#define OpenCL_API_hpp
#include "DeviceAPI.hpp"
#include "CL/cl.h"
#include <stdio.h> 



class OpenCL: public DeviceAPI{

private:
    int error;
	cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    
    cl_mem input;                       // device memory used for the input array
    cl_mem output;                      // device memory used for the output array
	void setup();


public:
    OpenCL();
    OpenCL(int blocksize);
    //void* allocate(unsigned int size);
	void allocate(void** ptr, unsigned int size);
    void copy_to_host(void * hostPointer, void * devicepointer, unsigned int size);
    void copy_to_device(void* devicePointer, void* hostPointer, unsigned int size);
    void deallocate(void* devicePointer);
	double getGPUElement(double * vec, unsigned int index);
	unsigned int getGPUElement(unsigned int * vec, unsigned int index);


    void cuda_sync_and_check(const char * caller);
    double sum_of_elements(double* vec,unsigned int size,unsigned int bufferedSize);
    double dotProduct(double * v1, double * v2, double * scratch, unsigned int size);
    void add_with_mult(double * a,double * b, double lambda, unsigned int size);//a = a + b* lambda
    
    void project_force(double* force,double *gradAVert,double * gradVVert, double scale,unsigned int size);
    void facet_to_vertex(double* vertexValue, double* facetValue,unsigned int* vertToFacet, unsigned int* vertIndexStart,unsigned int numVert);

    void area_gradient(double * gradAFacet,unsigned int* facets,double * vert,unsigned int numFacets);
    void volume_gradient(double * gradVFacet,unsigned int* facets,double * vert,unsigned int numFacets);

    void area(double * area, double * vert, unsigned int * facets, unsigned int numFacets);
    void volume(double * volume, double * vert, unsigned int * facets, unsigned int numFacets);


};





kernal void areaKernel(double * area, double * vert, unsigned int * facets, unsigned int numFacets);
kernal void volumeKernel(double * volume, double * vert, unsigned int * facets, unsigned int numFacets);
kernal void addTree(double * in, double * out);
kernal void addWithMultKernel(double *a ,double *b,double lambda, unsigned int size); // a += b * lambda
kernal void areaGradient(double* gradAFacet, unsigned int* facets,double* verts,unsigned int numFacets);
kernal void volumeGradient(double* gradVFacet, unsigned int* facets,double* verts,unsigned int numFacets);
kernal void facetToVertex(double* vertexValue, double* facetValue,unsigned int* vertToFacet, unsigned int* vertIndexStart,unsigned int numVert);
kernal void projectForce(double* force,double* gradAVert,double* gradVVert,double scale,unsigned int numEle);
kernal void elementMultiply(double* v1, double* v2, double* out, unsigned int size);

void vectorSub(double * v1, double * v2, double * vOut);
void vectorAdd(double * v1, double * v2, double * vOut);
void vecScale(double *v, double lambda);
void vecAssign(double *out, double *in,double lambda); // out  = in*lambda
void cross(double *a,double *b, double *c);
double dot(double *a, double *b, double *c);
double norm(double *a); 
int sign(double a);


    

#endif