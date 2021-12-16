#pragma once
#ifndef Mesh_hpp
#define Mesh_hpp

#include <math.h>
#include <stdio.h>
#include <string.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Kernals.hpp"
#include "Gradient.hpp"

// [x] Write DeviceMesh stuff that need to happen on the CPU 
// [x] write everything around the kernals for device mesh
// [x] write everything around the kernals for gradient
// [x] write kernals
// [x] write the __device__ util functions
// [x] get the correct size allocation for area and volume vectors (for add tree)
// [x] write the print for the mesh
// [x] write the optimizer 
// [x] write vector dot
// [x] write vector scale and subtract
// [x] GLOBAL constraints
// [x] let meshes constuct via pointer assignment (removes need for freind)
// [x] reproject
// [ ] write tests
class Gradient;


class Mesh {
private:

	unsigned int _numVert = 0;
	unsigned int _numFacets = 0;
	double* _vert = nullptr;
	unsigned int* _facets = nullptr;

    // helper function for file reading
    int getNumVertices(FILE* fp);
    int getNumFacets(FILE* fp);

public:
	Mesh();
	Mesh(const char*);
	Mesh(unsigned int,unsigned int,double *,unsigned int*);
	~Mesh();

	bool operator ==(const Mesh& rhs);


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
	bool print(const char*);
};

class DeviceMesh{
private:
    unsigned int _numVert = 0;
	unsigned int _numFacets = 0;
	double* _vert = nullptr;
	unsigned int* _facets = nullptr;

	// arrays holding the map from vertex to <facet, # in facet>
    unsigned int* _vertToFacet = nullptr; // the a list of facet indcies sorted by vertex
    unsigned int* _vertIndexStart = nullptr; // where the indcies in vertToFacet start for a vertex 


	double* _area = nullptr; // holds the area per facet
	//double* _areaSum = nullptr; // array for summing the area per facet

	double* _volume = nullptr; // holds the volume per facet
	//double* _volumeSum = nullptr; // array for summing the volume per facet

	cudaError_t _cudaStatus = cudaSetDevice(0);
	unsigned int _blockSize;
	unsigned int _bufferedSize;
	
public:
	DeviceMesh(Mesh*,unsigned int); //copies a Mesh over to the device 
	~DeviceMesh();

	Mesh copy_to_host();
	void decend_gradient(Gradient *,double);

	double volume();
	double area();
	double* check_area_on_facet();
	

	unsigned int get_numVert(){ return _numVert; }
	unsigned int get_numFacets(){ return _numFacets; }
	unsigned int get_blockSize(){ return _blockSize; }
	double* get_vert() { return _vert; }
	unsigned int* get_facets(){ return _facets; }
	unsigned int* get_vertToFacet(){ return _vertToFacet; }
	unsigned int* get_vertIndexStart(){ return _vertIndexStart; }
	


};




#endif