
#include <thrust/device_vector.h>

#include <Windowsnumerics.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>  
#include <string.h>
#include <vector>

#include "GPUShapeOptimizer.hpp"

#define BLOCKSIZE 2


int main()
{
    GPUShapeOptimizer myOpt = GPUShapeOptimizer("sphere.mesh");

    // read in mesh
    if (!myOpt.get_readSuccess()) {
        fprintf(stderr, "failed to read in mesh");
        return -1;
    }

    fprintf(stdout, "Read in mesh with %d vertices and %d faces\n", myOpt.get_meshSize(), myOpt.get_facetSize());
    bool checkMesh = false;
    if (checkMesh) {
        myOpt.printMesh();
    }


    
   
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
        addTree << <ceil((float)addNumBlock/ (BLOCKSIZE * 2)), BLOCKSIZE, BufferedSize / 2 * sizeof(float) >> > (dev_areaSum, dev_areaSum);
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



