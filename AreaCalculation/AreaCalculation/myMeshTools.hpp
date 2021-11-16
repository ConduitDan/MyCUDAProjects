#pragma once
#ifndef myMeshTools_hpp
#define myMeshTools_hpp


class myMeshTools {
private:
	unsigned int getNumFacets(FILE* fp);
	unsigned int getNumVertices(FILE* fp);

public:
	bool readInMesh(const char* fileName, float3** verts, unsigned int** facets, unsigned int* nVert, unsigned int* nFace);
	bool readInMesh(const char* fileName, Mesh* myMesh);



};