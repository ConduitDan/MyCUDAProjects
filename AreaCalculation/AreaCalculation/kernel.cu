﻿#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <stdio.h>

#define BLOCKSIZE 32

cudaError_t areaWithCuda(const float* vertices, const unsigned int  meshSize, const unsigned int* facets, \
    const unsigned int facetSize, float* areaPerFace, float* area);

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
        area[i] = (vertices[facets[i * 3] * 2] * vertices[facets[i * 3 + 1] * 2 + 1] /
            +vertices[facets[i * 3 + 1] * 2] * vertices[facets[i * 3 + 2] * 2 + 1] /
            +vertices[facets[i * 3 + 2] * 2] * vertices[facets[i * 3] * 2 + 1] /
            +vertices[facets[i * 3] * 2] * vertices[facets[i * 3 + 2] * 2 + 1] /
            +vertices[facets[i * 3 + 1] * 2] * vertices[facets[i * 3] * 2 + 1] /
            +vertices[facets[i * 3 + 2] * 2] * vertices[facets[i * 3 + 1] * 2 + 1]) / 2;
    }
}


__global__ void addTree(float* g_idata, float* g_odata, const unsigned int size)
{
    //https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x; // get the id of this thread
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];// g_idata[i]; // move the data over

   __syncthreads();
        // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
       __syncthreads();
    }
    __syncthreads();
    
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];

}

int main()
{
    const int meshSize = 8;
    const int facetSize = 6;
    const float vertices[meshSize * 2] = { 0,0,1,0,2,0,1,0.5,1,1.5,2,0,2,1,2,2 };
    //x---x---x
    // \  /\  /
    //  \/  \/
    //  x----x
    //  /\   /\ 
    // /  \ /  \
    //x----x----x
    const unsigned int facets[facetSize * 3] = { 0, 1, 3, \
                                        3, 4, 1, \
                                        1, 2, 4, \
                                        3, 5, 6, \
                                        3, 6, 4, \
                                        6, 4, 7 };
    float *areaPerFace = (float *) malloc(facetSize * sizeof(float));
    float area = 0;

    // Add vectors in parallel.
    cudaError_t cudaStatus = areaWithCuda(vertices, meshSize, facets, facetSize, areaPerFace, &area);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "areaWithCuda failed!");
        return 1;
    }

    printf("area on facets: \n");
    for (int i = 0; i < facetSize; i++) {
        printf("%d: %f\n", i, areaPerFace[i]);
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
cudaError_t areaWithCuda(const float* vertices, const unsigned int  meshSize, const unsigned int* facets, \
    const unsigned int facetSize, float * areaPerFace, float * area)
{
    float *dev_vertices = 0;
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

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_areaPerFace, facetSize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_areaSum, facetSize * sizeof(float)); // this should be facetSize/Num
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_vertices, 2 * meshSize * sizeof(float));
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
    cudaStatus = cudaMemcpy(dev_vertices, vertices, 2 * meshSize * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! vertices");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_facets, facets, 3 * facetSize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! facets");
        goto Error;
    }




    // Launch a kernel on the GPU with one thread for each element.
    areaKernel <<<ceil(facetSize/(double)BLOCKSIZE), BLOCKSIZE>>> (dev_areaPerFace, dev_vertices, dev_facets, facetSize);

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

    // now sum the result
    addTree <<<ceil(facetSize / (double)BLOCKSIZE /2.0), BLOCKSIZE >>> (dev_areaPerFace, dev_areaSum, facetSize);

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
