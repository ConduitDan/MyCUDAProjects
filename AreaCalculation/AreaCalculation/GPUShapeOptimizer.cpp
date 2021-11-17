
#include "GPUShapeOptimzer.hpp"


float3 GPUShapeOptimizer::operator*(const float alpha, const float3 v) { return make_float3(alpha * v.x, alpha * v.y, alpha * v.z); }

__global__ void GPUShapeOptimizer::areaKernel3d(float* area, const float* vertices, const unsigned int* facets, const int size)
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
        area[i] = abs(dx1 * (dy2 - dz2) + dx2 * (dz1 - dy1) + dy1 * dz2 - dz1 * dy2) / 2;
    }
    else {
        area[i] = 0;
    }
}
__device__ float3 GPUShapeOptimizer::cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, b.x * a.z - a.x * b.z, a.x * b.y - b.x * a.y);
}

__device__ float GPUShapeOptimizer::dot(float3 a, float3 b) {
    return float(a.x * b.x + a.y * b.y + a.z * b.z);
}

__device__ float GPUShapeOptimizer::norm(float3 a) {
    return sqrt(dot(a, a));
}
__device__ float3 GPUShapeOptimizer::sub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ float3 GPUShapeOptimizer::add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__global__ void GPUShapeOptimizer::areaKernel3dVectorized(float* area, const float3* vertices, const unsigned int* facets, const int size)
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
__global__ void GPUShapeOptimizer::areaGradVectorize(float3* delAreaFacet, const float3* vertices, const unsigned int* facets, const int size) {
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

__global__ void GPUShapeOptimizer::areaGradVertex(float3* delAreaVertex, const float3* delAreaFacet, const std::vector <std::tuple <unsigned int, unsigned int>>* vertex2facet[]) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //vertex2facet contains tuples of (facet, vertex# on facet)
    delAreaVertex[i] = make_float3(0.0, 0.0, 0.0);
    for (std::vector <std::tuple<unsigned int, unsigned int>>::iterator index = vertex2facet[i].begin(); index != vertex2facet[i].end(); index++) {
        delAreaVertex[i] = add(delAreaVertex[i], delAreaFacet[std::get<0>(*index) * 3 + std::get<1>(*index)]);
    }
}

__global__ void GPUShapeOptimizer::addTree(const float* g_idata, float* g_odata)
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


GPUShapeOptimizer::GPUShapeOptimizer(const char* fileName) {
    readSuccess = myMeshTools::readInMesh(fileName, &verts, &facets, &meshSize, &facetSize);

    // make the vertex to facet matrix
    vertex2facetMaker();
}

void GPUShapeOptimizer::vertex2facetMaker() {
    *vertex2facet = new std::vector < std::tuple<unsigned int, unsigned int>>[meshSize];
    for (int i = 0; i < numFacets; i++) {
        for (int j = 0; j < 3; j++) {
            (*vertex2facet)[facets[i * 3 + j]].push_back(std::tuple <unsigned int, unsigned int>{i, j});
        }
    }
}

void GPUShapeOptimizer::printMesh() {

    fprintf(stdout, "Vertices:\n");
    for (int i = 0; i < meshSize; i++) {
        fprintf(stdout, "%d: (%f, %f, %f)\n", i, vertices[i].x, vertices[i].y, vertices[i].z);
    }
    fprintf(stdout, "Faces:\n");
    for (int i = 0; i < facetSize; i++) {
        fprintf(stdout, "%d: (%d, %d, %d)\n", i, facets[i], facets[i + 1], facets[i + 2]);
    }

}