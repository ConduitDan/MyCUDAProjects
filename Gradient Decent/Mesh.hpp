#pragma once
#ifndef Mesh_hpp
#define Mesh_hpp

#include <stdio.h>
#include <string.h>


class Mesh {
private:

	unsigned int _numVert = 0;
	unsigned int _numFacets = 0;
	double* _vert = NULL;
	unsigned int* _facets = NULL;

    // helper function for file reading
    int getNumVertices(FILE* fp);
    int Mesh::getNumFacets(FILE* fp);


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
    unsigned int _numVert = 0;
	unsigned int _numFacets = 0;
	double* _vert = NULL;
	unsigned int* _facets = NULL;


public:
	DeviceMesh(Mesh); //copies a Mesh over to the device 
	~DeviceMesh();

	Mesh copy_to_host();
};

#endif