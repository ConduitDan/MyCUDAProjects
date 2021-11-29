
Skip to content
Pull requests
Issues
Marketplace
Explore
@ConduitDan
ConduitDan /
MyCUDAProjects
Private

1
0

    0

Code
Issues
Pull requests
Actions
Projects
Security
Insights

    Settings

MyCUDAProjects/AreaCalculation/AreaCalculation/kernel.cu
@ConduitDan
ConduitDan a method to find grad A per facet
Latest commit 1e9a981 7 days ago
History
1 contributor
482 lines (396 sloc) 17.6 KB
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <thrust/device_vector.h>

#include <Windowsnumerics.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>  
#include <string.h>
#include <vector>

#define BLOCKSIZE 2
int getNumVertices(FILE* fp);
int getNumFacets(FILE* fp);
bool readInMesh(const char* fileName, float3** verts, unsigned int** facets, unsigned int* nVert, unsigned int* nFace);
void vertex2facetMaker(std::vector < std::tuple<unsigned int, unsigned int>>** vertex2facet, unsigned int facets[], unsigned int meshSize, unsigned int numFacets);

float3 operator*(const float alpha, const float3 v) { return make_float3(alpha * v.x, alpha * v.y, alpha * v.z); }

cudaError_t areaWithCuda(float3* vertices, unsigned int  meshSize, unsigned int* facets, \
    unsigned int facetSize, float* areaPerFace, float* area);

__global__ void areaKernel(float *area, const float *vertices, const unsigned int * facets, const int size)
{
    // given a set of vertices and facet [v0,v1,v2](list of indeices of vertices belonging to a face) fill in what the area of that face is
    
    // formula is (x1*y2+x2*y3+x3*y1-y1*x2-y2*x3-y3*x1)/2 
    // NOTE THIS CAN BE DONE MORE IN PARALLEL
    // Check for vetorized instruction for cross product

    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // do i*3 because we have 3 vertcies per facet
    // do facets[]*2 becasue we have x and y positions
    if (i < size) {
        area[i] = abs(vertices[facets[i * 3] * 2] * (vertices[facets[i * 3 + 1] * 2 + 1] - vertices[facets[i * 3 + 2] * 2 + 1]) \
            + vertices[facets[i * 3 + 1] * 2] * (vertices[facets[i * 3 + 2] * 2 + 1] - vertices[facets[i * 3] * 2 + 1]) \
            + vertices[facets[i * 3 + 2] * 2] * (vertices[facets[i * 3] * 2 + 1] - vertices[facets[i * 3 + 1] * 2 + 1])) / 2;
    }
    else {
        area[i] = 0;
    }
}

__global__ void areaKernel3d(float* area, const float* vertices, const unsigned int* facets, const int size)
{
    // given a set of vertices and facet [v0,v1,v2](list of indeices of vertices belonging to a face) fill in what the area of that face is

    // formula is (x1*y2+x2*y3+x3*y1-y1*x2-y2*x3-y3*x1)/2 
    // NOTE THIS CAN BE DONE MORE IN PARALLEL
    // Check for vetorized instruction for cross product


    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // do i*3 because we have 3 vertcies per facet
    // do facets[]*2 becasue we have x and y positions
    float dx1 = vertices[facets[i * 3] * 3] - vertices[facets[i * 3 + 1] * 3];
    float dx2 = vertices[facets[i * 3 + 1] * 3] - vertices[facets[i * 3 + 2] * 3];
    float dy1 = vertices[facets[i * 3] * 3 + 1] - vertices[facets[i * 3 + 1] * 3 + 1];
    float dy2 = vertices[facets[i * 3 + 1] * 3 + 1] - vertices[facets[i * 3 + 2] * 3 + 1];
    float dz1 = vertices[facets[i * 3] * 3 + 2] - vertices[facets[i * 3 + 1] * 3 + 2];
    float dz2 = vertices[facets[i * 3 + 1] * 3 + 2] - vertices[facets[i * 3 + 2] * 3 + 2];
    if (i < size) {
        area[i] = abs(dx1*(dy2-dz2)+dx2*(dz1-dy1) + dy1*dz2 - dz1*dy2)/2;
    }
    else {
        area[i] = 0;
    }
}
__device__ float3 cross(float3 a,float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, b.x * a.z - a.x * b.z, a.x * b.y - b.x * a.y);
}

__device__ float dot(float3 a, float3 b) {
    return float(a.x * b.x + a.y * b.y + a.z * b.z);
}

__device__ float norm(float3 a) {
    return sqrt(dot(a, a));
}
__device__ float3 sub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ float3 add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__global__ void areaKernel3dVectorized(float* area, const float3* vertices, const unsigned int* facets, const int size)
{
    // given a set of vertices and facet [v0,v1,v2](list of indeices of vertices belonging to a face) fill in what the area of that face is

    // formula is (x1*y2+x2*y3+x3*y1-y1*x2-y2*x3-y3*x1)/2 
    // NOTE THIS CAN BE DONE MORE IN PARALLEL
    // Check for vetorized instruction for cross product


    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // do i*3 because we have 3 vertcies per facet
    // do facets[]*2 becasue we have x and y positions
    
    if (i < size) {
        area[i] = norm(cross(sub(vertices[facets[i]], vertices[facets[i + 1]]), sub(vertices[facets[i]], vertices[facets[i + 2]])));
    }
    else {
        area[i] = 0;
    }
}
__global__ void areaGradVectorize(float3* delAreaFacet, const float3* vertices, const unsigned int* facets, const int size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float3 s0 = sub(vertices[facets[i * 3 + 1]], vertices[facets[i * 3]]);
    float3 s1 = sub(vertices[facets[i * 3 + 2]], vertices[facets[i * 3 + 1]]);
    float3 s01 = cross(s0, s1);
    float snorm = norm(s01);
    float3 s010 = cross(s01, s0);
    float3 s011 = cross(s01, s1);
    // here is the tricky bit, cannot assume that there are no 2 facets which don't share a 1st point;
    // instead we could store these three 3vectors on the facet 
    // create the vertex -> facet map 
    // have each vertex go retrive its values and add them up
    // we might be able to take advatage of that matrix stucture to store data smartly
    
    delAreaFacet[i * 3] = 0.5 / snorm * s011;
    delAreaFacet[i * 3 + 2] = 0.5 / snorm * s010;
    delAreaFacet[i * 3 + 1] = -0.5 / snorm * cross(s010, s011);
}

__global__ void areaGradVertex(float3* delAreaVertex, const float3* delAreaFacet, const std::vector <std::tuple <unsigned int, unsigned int>>* vertex2facet[]) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //vertex2facet contains tuples of (facet, vertex# on facet)
    delAreaVertex[i] = make_float3(0.0, 0.0, 0.0);
    for (std::vector <std::tuple<unsigned int, unsigned int>>::iterator index = vertex2facet[i].begin(); index != vertex2facet[i].end(); index++) {
        delAreaVertex[i] = add(delAreaVertex[i], delAreaFacet[std::get<0>(*index) * 3 + std::get<1>(*index)]);
    }
}

__global__ void addTree(const float* g_idata, float* g_odata)
{
    //https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x; // get the id of this thread
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    //if (i + blockDim.x < size) {
    //printf("tid: %d\ti:%d\ti + blockDim.x:%d\tg_idata[i]:%f\tg_idata[i + blockDim.x]%f\n",tid,i, i + blockDim.x, g_idata[i] , g_idata[i + blockDim.x]);
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];// g_idata[i]; // move the data over
    //printf("tid: %d\ti:%d\ti + blockDim.x:%d\tg_idata[i]:%f\tg_idata[i + blockDim.x]%f\t sdata[tid]: %f\n", tid, i, i + blockDim.x, g_idata[i], g_idata[i + blockDim.x], sdata[tid]);
                                                      //}
   __syncthreads();
        // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
       __syncthreads();
    }
    __syncthreads();
    
//     write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];

 }

int main()
{

    // read in mesh

    unsigned int meshSize = 0;
    unsigned int facetSize = 0;
    float3* vertices = NULL;

    unsigned int * facets = NULL;
    bool checkMesh = false;

    bool readSuccess = readInMesh("sphere.mesh", &vertices, &facets, &meshSize, &facetSize);
    if (!readSuccess) {
        fprintf(stderr, "failed to read in mesh");
        return -1;
    }
    fprintf(stdout, "Read in mesh with %d vertices and %d faces\n", meshSize, facetSize);

    if (checkMesh) {
        fprintf(stdout, "Vertices:\n");
        for (int i = 0; i < meshSize; i++) {
            fprintf(stdout, "%d: (%f, %f, %f)\n", i, vertices[i].x, vertices[i].y, vertices[i].z);
        }
        fprintf(stdout, "Faces:\n");
        for (int i = 0; i < facetSize; i++) {
            fprintf(stdout, "%d: (%d, %d, %d)\n", i, facets[i], facets[i + 1], facets[i + 2]);
        }
    }

    std::vector < std::tuple<unsigned int, unsigned int>>* vertex2facet = NULL; 
    vertex2facetMaker(&vertex2facet, facets, meshSize, facetSize);

    
    float *areaPerFace = (float *) malloc(facetSize * sizeof(float));
    float area = 0;
    float areaCPU = 0;
    // Add vectors in parallel.
    cudaError_t cudaStatus = areaWithCuda(vertices, meshSize, facets, facetSize, areaPerFace, &area);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "areaWithCuda failed!");
        return 1;
    }

    printf("Facets:\n");
    for (int i = 0; i < facetSize; i++) {
        printf("%d: (%f, %f), (%f, %f) , (%f, %f)\n", i, vertices[facets[i * 3] * 2], vertices[facets[i * 3] * 2 + 1],\
            vertices[facets[i * 3 + 1] * 2], vertices[facets[i * 3 + 1] * 2 + 1],\
            vertices[facets[i * 3 + 2] * 2], vertices[facets[i * 3 + 2] * 2 + 1]);
    }

    printf("area on facets: \n");
    for (int i = 0; i < facetSize; i++) {
        //areaCPU = (vertices[facets[i * 3] * 2] * (vertices[facets[i * 3 + 1] * 2 + 1] - vertices[facets[i * 3 + 2] * 2 + 1]) \
            + vertices[facets[i * 3 + 1] * 2] * (vertices[facets[i * 3 + 2] * 2 + 1] - vertices[facets[i * 3] * 2 + 1]) \
            + vertices[facets[i * 3 + 2] * 2] * (vertices[facets[i * 3] * 2 + 1] - vertices[facets[i * 3 + 1] * 2 + 1])) / 2;
        printf("%d: GPU %f \n", i, areaPerFace[i]);
    }
    printf("Total Area: %f\n", area);


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t areaWithCuda(float3* vertices, unsigned int  meshSize, unsigned int* facets, \
    unsigned int facetSize, float * areaPerFace, float * area)
{
    float3 *dev_vertices = 0;
    unsigned int *dev_facets = 0;
    float *dev_areaPerFace = 0;
    float *dev_areaSum = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
    unsigned int BufferedSize = ceil(facetSize / (float)(2 * BLOCKSIZE)) * 2 * BLOCKSIZE;

    // Allocate GPU buffers for three vectors (two input, one output)    .
    // round up areaPerFace so that every thread in every block can assign and do something
    cudaStatus = cudaMalloc((void**)&dev_areaPerFace, BufferedSize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    unsigned int numSum = BufferedSize / BLOCKSIZE / 2;
    cudaStatus = cudaMalloc((void**)&dev_areaSum, numSum * sizeof(float)); // this should be facetSize/Num
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_vertices, meshSize * sizeof(float3));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_facets, 3 * facetSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    //    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice)
    cudaStatus = cudaMemcpy(dev_vertices, vertices, meshSize * sizeof(float3), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! vertices\n");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_facets, facets, 3 * facetSize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! facets\n");
        goto Error;
    }


    unsigned int areaNumBlock = ceil(BufferedSize / (float)BLOCKSIZE);

    // Launch a kernel on the GPU with one thread for each element.
    areaKernel3dVectorized <<<areaNumBlock, BLOCKSIZE>>> (dev_areaPerFace, dev_vertices, dev_facets, facetSize);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "areaKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error %s after launching areaKernel!\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    unsigned int addNumBlock = ceil(BufferedSize / (float)BLOCKSIZE / 2.0);
    // now sum the result
    addTree << <addNumBlock, BLOCKSIZE, BufferedSize / 2 * sizeof(float) >> > (dev_areaPerFace, dev_areaSum);
    for (int i = addNumBlock; i > 1; i /= (BLOCKSIZE * 2)) {
        addTree <<<ceil((float)addNumBlock/ (BLOCKSIZE * 2)), BLOCKSIZE, BufferedSize / 2 * sizeof(float) >>> (dev_areaSum, dev_areaSum);
    }
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error %s after launching addKernel!\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }






    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(areaPerFace, dev_areaPerFace, facetSize * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! area per facet\n");
        goto Error;
    }
    cudaStatus = cudaMemcpy(area, dev_areaSum,sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! area\n");
        goto Error;
    }


Error:


    cudaFree(dev_vertices);
    cudaFree(dev_facets);
    cudaFree(dev_areaPerFace);
    cudaFree(dev_areaSum);

    return cudaStatus;
}

int getNumVertices(FILE* fp) {

    int numVertices = 0;
    char line[50];
    fscanf(fp, "%s\n", line);
    if (strcmp(line,"vertices")) {
        fprintf(stderr, "File didn't start with 'vertices'\n");
        return -1;

    }
    fgets(line, 50, fp); // eat the new line
    fgets(line, 50, fp); // read line 1
    numVertices++;
    while (strcmp(line ,"\n")&& !feof(fp)) {
        numVertices++;
        fgets(line, 50, fp); // read line n
    }
return numVertices;

}
int getNumFacets(FILE* fp) {
    int numFaces = 0;
    char line[50];
    fscanf(fp, "%s\n", line);
    while (strcmp(line,"faces")) {
        fscanf(fp, "%s\n", line);
        if (feof(fp)) {
            fprintf(stderr, "File had no faces\n");
            return -1;
        }
    }
    fgets(line, 50, fp); // eat the new line
    fgets(line, 50, fp); // read line 1
    numFaces++;
    while (strcmp(line,"\n")&& !feof(fp)) {
        numFaces++;
        fgets(line, 50, fp); // read line 1

    }
    return numFaces;

}


bool readInMesh(const char* fileName, float3** verts, unsigned int** facets, unsigned int* nVert, unsigned int* nFace) {
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
    if (strcmp(sectionHeader ,"vertices")) {
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
    while (strcmp(sectionHeader ,"faces")) {
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

void vertex2facetMaker(std::vector < std::tuple<unsigned int, unsigned int>>** vertex2facet, unsigned int facets[],unsigned int meshSize,unsigned int numFacets) {
    *vertex2facet = new std::vector < std::tuple<unsigned int, unsigned int>>[meshSize];
    for (int i = 0; i < numFacets; i++) {
        for (int j = 0; j < 3; j++) {
            (*vertex2facet)[facets[i * 3 + j]].push_back(std::tuple <unsigned int, unsigned int>{i, j});
        }
    }
}
