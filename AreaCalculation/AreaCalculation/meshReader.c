#include "meshReader.h"



int getNumVertices(FILE* fp) {
	int numVertices = 0;
	char* line[50];
	fscanf(fp, "%s\n", *line);
	if (line != "vertices") {
		return -1;
	}
	fgets(line, 50, fp); // eat the new line
	fgets(line, 50, fp); // read line 1
	numVertices++;
	while (line != "\n") {
		numVertices++;
	}
	return numVertices;

}
int getNumFacets(FILE* fp) {
	int numFaces = 0;
	char* line[50];
	fscanf(fp, "%s\n", *line);
	while (line != "faces") {
		fscanf(fp, "%s\n", *line);
		if (feof(fp)) {
			return -1;
		}
	}
	fgets(line, 50, fp); // eat the new line
	fgets(line, 50, fp); // read line 1
	numFaces++;
	while (line != "\n") {
		numFaces++;
	}

}


bool readInMesh(const char* fileName, double* verts, unsigned int* facets,unsigned int * nVert, unsigned int * nFace) {
	FILE* fp; 
	char* line = NULL;
	size_t len = 0;
	char* sectionHeader[50];

	int numAssigned = 0;

	fp = fopen(fileName, "r");
	if (fp == NULL)
		return false;

	*nVert = getNumVertices(fp);
	*nFace = getNumFacets(fp);

	verts = malloc(*nVert * 3 * sizeof(double)); // [x0; y0; z0; x1; y1;.... ]
	facets = malloc(*nFace * 3 * sizeof(unsigned int));// [a0; b0; c0; a1;b1;c1;...]


	rewind(fp); // rewind the file to the beginning
	// make sure the first line say vertices

	fscanf(fp, "%s\n", *sectionHeader);
	if (sectionHeader != "vertices") {
		return false;
	}
	// get past the empty line
	
	for (int i = 0; i < *nVert; i++) {
		numAssigned = fscanf(fp, "%*d %f %f %f\n", verts[i*3], verts[i * 3 + 1], verts[i * 3 + 2]);
		if (numAssigned < 3)
			return false;

	}

	fscanf(fp, "%*d");
	fscanf(fp, "%s\n", *sectionHeader);
	while (sectionHeader != "faces") {
		fscanf(fp, "%s\n", *sectionHeader);
		if (feof(fp)) {
			return false;
		}
	}

	for (int i = 0; i < *nFace; i++) {
		numAssigned = fscanf(fp, "%*d %d %d %d\n", facets[i * 3], facets[i * 3 + 1], facets[i * 3 + 2]);
		if (numAssigned < 3)
			return false;

	}
	return true;

}
