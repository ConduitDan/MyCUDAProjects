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

#define PROGRAM_LIST_VAR {areaKernelSTR,\
						volumeKernelSTR,\
						addTreeSTR,\
						addWithMultKernelSTR,\
						areaGradientSTR,\
						volumeGradientSTR,\
						facetToVertexSTR,\
						projectForceSTR,\
						elementMultiplySTR}

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
    cl_program program_list[NUMBER_OF_PROGRAMS];// compute program
	cl_kernel kernel_list[NUMBER_OF_PROGRAMS];	// compute kernel
	const char* programNames[NUMBER_OF_PROGRAMS] = PROGRAM_LIST;
	const char* programVarNames[NUMBER_OF_PROGRAMS] = {};
	/*{areaKernelSTR,\
						volumeKernelSTR,\
						addTreeSTR,\
						addWithMultKernelSTR,\
						areaGradientSTR,\
						volumeGradientSTR,\
						facetToVertexSTR,\
						projectForceSTR,\
						elementMultiplySTR};*/


    
	void setup();
	void one_add_tree(UniqueDevicePtr<double>* vec,size_t local, size_t global);

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





// kernal void areaKernel(double * area, double * vert, unsigned int * facets, unsigned int numFacets);
// kernal void volumeKernel(double * volume, double * vert, unsigned int * facets, unsigned int numFacets);
// kernal void addTree(double * in, double * out);
// kernal void addWithMultKernel(double *a ,double *b,double lambda, unsigned int size); // a += b * lambda
// kernal void areaGradient(double* gradAFacet, unsigned int* facets,double* verts,unsigned int numFacets);
// kernal void volumeGradient(double* gradVFacet, unsigned int* facets,double* verts,unsigned int numFacets);
// kernal void facetToVertex(double* vertexValue, double* facetValue,unsigned int* vertToFacet, unsigned int* vertIndexStart,unsigned int numVert);
// kernal void projectForce(double* force,double* gradAVert,double* gradVVert,double scale,unsigned int numEle);
// kernal void elementMultiply(double* v1, double* v2, double* out, unsigned int size);

void vectorSub(double * v1, double * v2, double * vOut);
void vectorAdd(double * v1, double * v2, double * vOut);
void vecScale(double *v, double lambda);
void vecAssign(double *out, double *in,double lambda); // out  = in*lambda
void cross(double *a,double *b, double *c);
double dot(double *a, double *b, double *c);
double norm(double *a); 
int sign(double a);


    

#endif