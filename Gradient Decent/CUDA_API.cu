// Kernals.cu
// here is where all the device and global functions live

#include "CUDA_API.hpp"


// make this thing a singleton

CUDA::CUDA():DeviceAPI(256){
    _cudaStatus = cudaSetDevice(0);
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }
}
CUDA::CUDA(int blockSizeIn):DeviceAPI(blockSizeIn){
	printf("Setting up GPU with blocksize %d\n",blockSizeIn);
    _cudaStatus = cudaSetDevice(0);
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }
}



void CUDA::allocate(void** ptr, unsigned int size){


    _cudaStatus = cudaMalloc((void**)ptr,(int) size);
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!%s\n",cudaGetErrorString(_cudaStatus));
    }
}
void CUDA::deallocate(void* devicePointer){
	if (devicePointer) {
		_cudaStatus = cudaFree(devicePointer);
	}
	if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaFree failed! %s\n",cudaGetErrorString(_cudaStatus));
		throw "oops";
    }

}


void CUDA::copy_to_device(void* devicePointer, void* hostPointer, unsigned int size){
    _cudaStatus = cudaMemcpy(devicePointer, hostPointer, size, cudaMemcpyHostToDevice);
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! %s\n",cudaGetErrorString(_cudaStatus));
		throw "oops";
    }
}

void CUDA::copy_to_host(void * hostPointer, void * devicepointer, unsigned int size){
    _cudaStatus = cudaMemcpy(hostPointer, devicepointer, size, cudaMemcpyDeviceToHost);
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! %s\n",cudaGetErrorString(_cudaStatus));
		throw "oops";
    }
}


void CUDA::project_force(UniqueDevicePtr<double>* force,UniqueDevicePtr<double>* gradAVert,UniqueDevicePtr<double>* gradVVert, double scale,unsigned int size){
    unsigned int numberOfBlocks = ceil(size / (float) blockSize);
    projectForce<<<numberOfBlocks, blockSize>>>((double*)force->get(),(double*)gradAVert->get(),(double*)gradVVert->get(),scale,size);
    cuda_sync_and_check("project to force");

}
void CUDA::facet_to_vertex(UniqueDevicePtr<double>* vertexValue, UniqueDevicePtr<double>* facetValue,UniqueDevicePtr<unsigned int>* vertToFacet, UniqueDevicePtr<unsigned int>* vertIndexStart,unsigned int numVert){
    unsigned int numberOfBlocks = ceil(numVert / (float) blockSize);
    facetToVertex<<<numberOfBlocks, blockSize>>>((double*)vertexValue->get(),\
												 (double*)facetValue->get(),\
												 (unsigned int*)vertToFacet->get(),\
												 (unsigned int*)vertIndexStart->get(),numVert);
    cuda_sync_and_check("facet_to_vertex");

}



void CUDA::area_gradient(UniqueDevicePtr<double>* gradAFacet,UniqueDevicePtr<unsigned int>* facets,UniqueDevicePtr<double>* vert,unsigned int numFacets){
    unsigned int numberOfBlocks = ceil(numFacets / (float) blockSize);
    areaGradient<<<numberOfBlocks, blockSize>>>((double*)gradAFacet->get(),(unsigned int*)facets->get(),(double*)vert->get(),numFacets);
    cuda_sync_and_check("GradA");


}
void CUDA::volume_gradient(UniqueDevicePtr<double>* gradVFacet,UniqueDevicePtr<unsigned int>* facets,UniqueDevicePtr<double>* vert,unsigned int numFacets){
    unsigned int numberOfBlocks = ceil(numFacets / (float) blockSize);
    volumeGradient<<<numberOfBlocks, blockSize>>>((double*)gradVFacet->get(), (unsigned int*)facets->get(), (double*)vert->get(), numFacets);
    cuda_sync_and_check("GradV");

}


double CUDA::sum_of_elements(UniqueDevicePtr<double>* vec,unsigned int size,unsigned int bufferedSize){

    double out;

    // do the reduction each step sums blockSize*2 number of elements
    unsigned int numberOfBlocks = ceil(size / (float) blockSize / 2.0);
    // printf("AddTree with %d blocks,  of blocks size %d, for %d total elements\n",numberOfBlocks,blockSize,_bufferedSize);
    
    addTree<<<numberOfBlocks, blockSize, blockSize  * sizeof(double) >>> ((double*)vec->get(), (double*)vec->get());
    cuda_sync_and_check("sum of elements");


    if (numberOfBlocks>1){
        for (int i = numberOfBlocks; i > 1; i /= (blockSize * 2)) {
            addTree<<<ceil((float)numberOfBlocks/ (blockSize * 2)), blockSize, blockSize * sizeof(double)>>> ((double*)vec->get(), (double*)vec->get());
            cuda_sync_and_check("sum of elements");
        } 
    }

    // copy the 0th element out of the vector now that it contains the sum
    copy_to_host(&out, vec->get(),sizeof(double));
  

    return out;

}

void CUDA::cuda_sync_and_check(const char * caller){
    _cudaStatus = cudaGetLastError();
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s. From %s\n", cudaGetErrorString(_cudaStatus),caller);
        throw "Kernel Launch Failure";
    }
    // check that the kernal didn't throw an error
    _cudaStatus = cudaDeviceSynchronize();
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error %s after launching Kernel %s!\n", cudaGetErrorString(_cudaStatus),caller);
        throw "Kernel Failure";
    }

}



double CUDA::dotProduct(UniqueDevicePtr<double>* v1, UniqueDevicePtr<double>* v2, UniqueDevicePtr<double>* scratch, unsigned int size){

    // first multiply
    unsigned int numberOfBlocks = ceil(size / (float) blockSize);

    elementMultiply<<<numberOfBlocks,blockSize>>>((double*)v1->get(),(double*)v2->get(), (double*)scratch->get(),size);
    cuda_sync_and_check("Element Multiply");
    unsigned int bufferedSize = ceil(size/(2.0*blockSize))*2 *blockSize;
    //now sum
    double out = sum_of_elements(scratch,size, bufferedSize);
	cuda_sync_and_check("sum_of_elments");

    // clear the scratch
    //cudaMemset(scratch->get(),0.0,sizeof(double)*bufferedSize);
	cuda_sync_and_check("memset");

    return out;


}


void CUDA::area(UniqueDevicePtr<double>* area, UniqueDevicePtr<double>* vert, UniqueDevicePtr<unsigned int>* facets, unsigned int numFacets){
    unsigned int numberOfBlocks = ceil(numFacets / (float) blockSize);
    areaKernel<<<numberOfBlocks,blockSize>>>((double*)area->get(),(double*) vert->get(), (unsigned int*) facets->get(), numFacets);
    cuda_sync_and_check("area");

}
void CUDA::volume(UniqueDevicePtr<double>* volume, UniqueDevicePtr<double>* vert, UniqueDevicePtr<unsigned int>* facets, unsigned int numFacets){
    unsigned int numberOfBlocks = ceil(numFacets / (float) blockSize);
    volumeKernel<<<numberOfBlocks,blockSize>>>((double*)volume->get(),(double*) vert->get(), (unsigned int*) facets->get(), numFacets);
    cuda_sync_and_check("area");

}
void CUDA::add_with_mult(UniqueDevicePtr<double>* a,UniqueDevicePtr<double>* b, double lambda, unsigned int size){
    unsigned int numberOfBlocks = ceil(size / (float) blockSize);
    addWithMultKernel<<<numberOfBlocks,blockSize>>>((double*)a->get(),(double*)b->get(),lambda,size);
}

double CUDA::getGPUElement(double * vec, unsigned int index){
	double out;
	_cudaStatus = cudaMemcpy(&out, vec + index ,sizeof(double), cudaMemcpyDeviceToHost);
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! area\n");
    throw;
    }

    return out;
}

unsigned int CUDA::getGPUElement(unsigned int * vec, unsigned int index){
	unsigned int out;
	_cudaStatus = cudaMemcpy(&out, vec + index ,sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! area\n");
    throw;
    }

    return out;
}
