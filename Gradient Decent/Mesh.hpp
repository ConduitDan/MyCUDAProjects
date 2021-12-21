#pragma once
#ifndef Mesh_hpp
#define Mesh_hpp

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <memory>
#include "Kernals.hpp"
#include "Gradient.hpp"

//#include "cuda.h"
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

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

	DeviceAPI * GPU = CUDA::Instance(); 

	using unique_device_double_ptr = std::unique_ptr<double,decltype(GPU->get_deallocate())>;
	using unique_device_uint_ptr = std::unique_ptr<unsigned int,decltype(GPU->get_deallocate())>;

	unique_device_double_ptr _vert;
	unique_device_uint_ptr _facets;

	// arrays holding the map from vertex to <facet, # in facet>
    unique_device_uint_ptr _vertToFacet; // the a list of facet indcies sorted by vertex
    unique_device_uint_ptr _vertIndexStart; // where the indcies in vertToFacet start for a vertex 


	unique_device_double_ptr _area; // holds the area per facet
	

	unique_device_double_ptr _volume; // holds the volume per facet
	//double* _volumeSum = nullptr; // array for summing the volume per facet

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
	double* get_vert() { return _vert.get(); }
	unsigned int* get_facets(){ return _facets.get(); }
	unsigned int* get_vertToFacet(){ return _vertToFacet.get(); }
	unsigned int* get_vertIndexStart(){ return _vertIndexStart.get(); }
	
};




#endif