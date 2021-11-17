#pragma once
#ifndef Mesh_hpp
#define Mesh_hpp

#include <Windowsnumerics.h>


class Mesh {
private:

	unsigned int _numVert = 0;
	unsigned int _numFacets = 0;
	float3* _vert = NULL;
	unsigned int* _facets = NULL;

public:
	Mesh();
	Mesh(const char*);
	~Mesh();

	// getters
	unsigned int get_numVert(){ return _numVert; }
	unsigned int get_numFacets(){ return _numFacets; }
	float3* get_vert() { return vert; }
	unsigned int* get_facets(){ return facets; }

	// setters
	void set_numVert(unsigned int numVert) { _numVert = numVert; }
	void get_numFacets(unsigned int numFacets) { _numFacets = numFacets; }
	void get_vert(float3* vert) { _vert = vert }
	void get_facets(unsigned int* facets){ return _facets = facets }

	void load_mesh_from_file(const char*);
};

class DeviceMesh (Mesh){
private:


public:
	DeviceMesh(Mesh); //copies a Mesh over to the device 
	~DeviceMesh();

	Mesh copy_to_host();
};