#pragma once
#ifndef OpenCL_API_hpp
#define OpenCL_API_hpp


#define PROGRAM_LIST {"areaKernel",\
						"volumeKernel",\
						"addTree",\
						"addWithMultKernel",\
						"areaGradient",\
						"volumeGradient",\
						"facetToVertex",\
						"projectForce",\
						"elementMultiply"}

#define AREA 0
#define VOLUME 1
#define ADDTREE 2
#define ADDWITHMULT 3
#define AREAGRAD 4
#define VOLUMEGRAD 5
#define FACETTOVERTEX 6
#define PROJECTFORCE 7
#define ELEMENTMULT 8

#define NUMBER_OF_PROGRAMS 9


#define CL_TARGET_OPENCL_VERSION 220
#define __CL_ENABLE_EXCEPTIONS
#include "DeviceAPI.hpp"
#include "CL/cl.h"
#include "kernalfile.hpp"
#include <stdio.h> 
#include <math.h>





class OpenCL: public DeviceAPI{

private:
    int error;
	cl_device_id device_id;             		// compute device id 
    cl_context context;                 		// compute context
    cl_command_queue commands;          		// compute command queue
    cl_program program;// compute program
	cl_kernel kernel_list[NUMBER_OF_PROGRAMS];	// compute kernel
	const char* programNames[NUMBER_OF_PROGRAMS] = PROGRAM_LIST;
	const char* sourcepath = "openCLKernels.cl";

    
	void setup();
	void one_add_tree(UniqueDevicePtr<double>* vec,size_t local, size_t global);
	const char* getErrorString(cl_int error);
	void checkSuccess(const char * caller);
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

	double sum_of_elements(UniqueDevicePtr<double>* vec,unsigned int size,unsigned int bufferedSize);
    double dotProduct(UniqueDevicePtr<double>* v1, UniqueDevicePtr<double>* v2, UniqueDevicePtr<double>* scratch, unsigned int size);
    void add_with_mult(UniqueDevicePtr<double>* a,UniqueDevicePtr<double>* b, double lambda, unsigned int size);//a = a + b* lambda
    
	void project_force(UniqueDevicePtr<double>* force,UniqueDevicePtr<double>* gradAVert,UniqueDevicePtr<double>* gradVVert, double scale,unsigned int size);
    void facet_to_vertex(UniqueDevicePtr<double>* vertexValue, UniqueDevicePtr<double>* facetValue,UniqueDevicePtr<unsigned int>* vertToFacet, UniqueDevicePtr<unsigned int>* vertIndexStart,unsigned int numVert);
    
	void area_gradient(UniqueDevicePtr<double>* gradAFacet,UniqueDevicePtr<unsigned int>* facets,UniqueDevicePtr<double>* vert,unsigned int numFacets);
    void volume_gradient(UniqueDevicePtr<double>* gradVFacet,UniqueDevicePtr<unsigned int>* facets,UniqueDevicePtr<double>* vert,unsigned int numFacets);
    
	void area(UniqueDevicePtr<double>* area, UniqueDevicePtr<double>* vert, UniqueDevicePtr<unsigned int>* facets, unsigned int numFacets);
    void volume(UniqueDevicePtr<double>* volume, UniqueDevicePtr<double>* vert, UniqueDevicePtr<unsigned int>* facets, unsigned int numFacets);


};
    
#endif