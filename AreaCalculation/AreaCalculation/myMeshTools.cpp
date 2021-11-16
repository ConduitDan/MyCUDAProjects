#include "myMeshTools.hpp"


int myMeshTools::getNumVertices(FILE* fp) {

    int numVertices = 0;
    char line[50];
    fscanf(fp, "%s\n", line);
    if (strcmp(line, "vertices")) {
        fprintf(stderr, "File didn't start with 'vertices'\n");
        return -1;

    }
    fgets(line, 50, fp); // eat the new line
    fgets(line, 50, fp); // read line 1
    numVertices++;
    while (strcmp(line, "\n") && !feof(fp)) {
        numVertices++;
        fgets(line, 50, fp); // read line n
    }
    return numVertices;

}



int myMeshTools::getNumFacets(FILE* fp) {
    int numFaces = 0;
    char line[50];
    fscanf(fp, "%s\n", line);
    while (strcmp(line, "faces")) {
        fscanf(fp, "%s\n", line);
        if (feof(fp)) {
            fprintf(stderr, "File had no faces\n");
            return -1;
        }
    }
    fgets(line, 50, fp); // eat the new line
    fgets(line, 50, fp); // read line 1
    numFaces++;
    while (strcmp(line, "\n") && !feof(fp)) {
        numFaces++;
        fgets(line, 50, fp); // read line 1

    }
    return numFaces;

}


bool myMeshTools::readInMesh(const char* fileName, Mesh mymesh) {

}


bool myMeshTools::readInMesh(const char* fileName, float3** verts, unsigned int** facets, unsigned int* nVert, unsigned int* nFace) {
    FILE* fp;
    char* line = NULL;
    size_t len = 0;
    char sectionHeader[50];

    int numAssigned = 0;

    fp = fopen(fileName, "r");
    if (fp == NULL) {
        fprintf(stderr, "Could not open file\n");
        return false;
    }

    *nVert = getNumVertices(fp);
    *nFace = getNumFacets(fp);

    *verts = (float3*)malloc(*nVert * sizeof(float3)); // [x0; y0; z0; x1; y1;.... ]
    *facets = (unsigned int*)malloc(*nFace * 3 * sizeof(unsigned int));// [a0; b0; c0; a1;b1;c1;...]


    rewind(fp); // rewind the file to the beginning
    // make sure the first line say vertices

    fscanf(fp, "%s\n", sectionHeader);
    if (strcmp(sectionHeader, "vertices")) {
        fprintf(stderr, "File didn't start with 'vertices'\n");
        return false;
    }
    // get past the empty line

    float tmp0, tmp1, tmp2;
    for (int i = 0; i < *nVert; i++) {
        numAssigned = fscanf(fp, "%*d %f %f %f\n", &tmp0, &tmp1, &tmp2);

        if (numAssigned < 3) {
            fprintf(stderr, "bad file format\n");
            return false;
        }
        (*verts)[i] = make_float3(tmp0, tmp1, tmp2);
    }

    fscanf(fp, "%*d");
    fscanf(fp, "%s\n", sectionHeader);
    while (strcmp(sectionHeader, "faces")) {
        fscanf(fp, "%s\n", sectionHeader);
        if (feof(fp)) {
            fprintf(stderr, "couldn't find faces\n");
            return false;
        }
    }

    for (int i = 0; i < *nFace; i++) {
        numAssigned = fscanf(fp, "%*d %d %d %d\n", (*facets) + i * 3, (*facets) + i * 3 + 1, (*facets) + i * 3 + 2);
        if (numAssigned < 3) {
            fprintf(stderr, "bad file format for faces\n");
            return false;
        }
        (*facets)[i * 3] --;
        (*facets)[i * 3 + 1] --;
        (*facets)[i * 3 + 2] --;

    }
    return true;
}