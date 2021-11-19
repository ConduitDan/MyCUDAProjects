#pragma once
#ifndef Mesh_hpp
#define Mesh_hpp

#include <math.h>
#include <stdio.h>
#include <string.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

// [x] Write DeviceMesh stuff that need to happen on the CPU 
// [x] write everything around the kernals for device mesh
// [ ] write everything around the kernals for gradient
// [ ] write kernals
// [ ] write the __device__ util functions
// [ ] write the print for the mesh
// [ ] write the optimizer class

class Mesh {
private:

	unsigned int _numVert = 0;
	unsigned int _numFacets = 0;
	double* _vert = nullptr;
	unsigned int* _facets = nullptr;

    // helper function for file reading
    int getNumVertices(FILE* fp);
    int Mesh::getNumFacets(FILE* fp);

	friend class DeviceMesh; // so device meshes can create meshs via copy

public:
	Mesh();
	Mesh(const char*);
	~Mesh();

	// getters
	unsigned int get_numVert(){ return _numVert; }
	unsigned int get_numFacets(){ return _numFacets; }
	double* get_vert() { return _vert; }
	unsigned int* get_facets(){ return _facets; }

	// setters
	//void set_numVert(unsigned int numVert) { _numVert = numVert; }
	//void get_numFacets(unsigned int numFacets) { _numFacets = numFacets; }
	//void get_vert(float3* vert) { _vert = vert }
	//void get_facets(unsigned int* facets){ return _facets = facets }

	bool load_mesh_from_file(const char*);
};

class DeviceMesh{
private:
    unsigned int _numVert = 0;
	unsigned int _numFacets = 0;
	double* _vert = nullptr;
	unsigned int* _facets = nullptr;

	// arrays holding the map from vertex to <facet, # in facet>
    unsigned int* _vertToFacet;
    unsigned int* _vertIndexStart;


	double* _area = nullptr; // holds the area per facet
	double* _areaSum = nullptr; // array for summing the area per facet

	double* _volume = nullptr; // holds the volume per facet
	double* _volumeSum = nullptr; // array for summing the volume per facet

    cudaError_t _cudaStatus;
	unsigned int _blockSize;
	
	double sum_of_elements(double* ,unsigned int);
	void cuda_sync_and_check();

public:
	DeviceMesh(Mesh,unsigned int); //copies a Mesh over to the device 
	~DeviceMesh();

	Mesh copy_to_host();
	void decend_gradient(Gradient*,double);

	double volume();
	double area();

	unsigned int get_numVert(){ return _numVert; }
	unsigned int get_numFacets(){ return _numFacets; }
	unsigned int get_blockSize(){ return _blockSize; }
	double* get_vert() { return _vert; }
	unsigned int* get_facets(){ return _facets; }
	unsigned int* get_vertToFacet(){ return _vertToFacet; }
	unsigned int* get_vertIndexStart(){ return _vertIndexStart; }


};

class Gradient{
private:
	DeviceMesh * _myMesh;

	double *_gradAFacet = nullptr;
	double *_gradAVert = nullptr;

	double *_gradVFacet = nullptr;
	double *_gradVVert = nullptr;

	double *_force = nullptr;

	cudaError_t _cudaStatus;


	void calc_gradA();
	void calc_gradV();
	void facet_to_vertex(double*, double*);
	void project_to_force();
	void cuda_sync_and_check();



public:
	Gradient(DeviceMesh*);
	~Gradient();
	void calc_force();

	double* get_force(){return _force;}
};




#endif