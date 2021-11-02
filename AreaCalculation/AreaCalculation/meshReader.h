#ifndef meshReader_h
#define meshReader_h
#ifdef __cplusplus 
extern "C" {
#endif
    /* Declarations of this file */
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>  

int getNumVertices(FILE *fp);
int getNumFacets(FILE *fp);
bool readInMesh(const char* fileName, double* verts, unsigned int* facets, unsigned int* nVert, unsigned int* nFace);

#ifdef __cplusplus
}

#endif

#endif // !meshReader_h





