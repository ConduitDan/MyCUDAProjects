//OpenCLKernals.cl

const char *areaKernel = "\n"\
"__kernal void area(__global double * area, __global double * vert, __global unsigned int * facets,__global unsigned int numFacets){\n"\
"    int i = blockDim.x * blockIdx.x + get_local_id();\n"\
"    double r10[3];\n"\
"   double r21[3];\n"\
"    double S[3];\n"\
"\n"\
"    if (i < numFacets) {\n"\
"        vectorSub(&vert[facets[i*3+1]*3], &vert[facets[i*3]*3],r10);\n"\
"        vectorSub(&vert[facets[i*3+2]*3], &vert[facets[i*3+1]*3],r21);  \n"\  
"        cross(r10, r21,S);\n"\
"        area[i] = norm(S)/2;\n"\
"    }\n"\
"    else {\n"\
"        area[i] = 0;\n"\
"    }\n";



kernal void areaKernel(double * area, double * vert, unsigned int * facets, unsigned int numFacets);
kernal void volumeKernel(double * volume, double * vert, unsigned int * facets, unsigned int numFacets);
kernal void addTree(double * in, double * out);
kernal void addWithMultKernel(double *a ,double *b,double lambda, unsigned int size); // a += b * lambda
kernal void areaGradient(double* gradAFacet, unsigned int* facets,double* verts,unsigned int numFacets);
kernal void volumeGradient(double* gradVFacet, unsigned int* facets,double* verts,unsigned int numFacets);
kernal void facetToVertex(double* vertexValue, double* facetValue,unsigned int* vertToFacet, unsigned int* vertIndexStart,unsigned int numVert);
kernal void projectForce(double* force,double* gradAVert,double* gradVVert,double scale,unsigned int numEle);
kernal void elementMultiply(double* v1, double* v2, double* out, unsigned int size);
