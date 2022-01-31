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

CUDA::~CUDA(){
	cudaDeviceReset();
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
	dim3 numberOfBlocks3D( numberOfBlocks, 1, 1 );
	dim3 blockSize3D( blockSize, 3, 1 );
    facetToVertex<<<numberOfBlocks3D, blockSize3D>>>((double*)vertexValue->get(),\
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
void CUDA::area_gradient2(UniqueDevicePtr<double>* gradAVert,UniqueDevicePtr<unsigned int>* facets,UniqueDevicePtr<double>* vert,unsigned int numFacets,unsigned int numVert){
	// mem set and call kernal
    cudaMemset(gradAVert->get(),0.0,sizeof(double)*numVert*3);
	cuda_sync_and_check("memset");
	unsigned int numberOfBlocks = ceil(numFacets / (float) blockSize);
    areaGradient2<<<numberOfBlocks, blockSize>>>((double*)gradAVert->get(), (unsigned int*)facets->get(), (double*)vert->get(),numFacets);
    cuda_sync_and_check("GradA");
//(double* gradVVert, unsigned int* facets,double* verts,unsigned int numFacets)
}
void CUDA::volume_gradient2(UniqueDevicePtr<double>* gradVVert,UniqueDevicePtr<unsigned int>* facets,UniqueDevicePtr<double>* vert,unsigned int numFacets,unsigned int numVert){
    cudaMemset(gradVVert->get(),0.0,sizeof(double)*numVert*3);
	cuda_sync_and_check("memset");

    unsigned int numberOfBlocks = ceil(numFacets / (float) blockSize);
    volumeGradient2<<<numberOfBlocks, blockSize>>>((double*)gradVVert->get(), (unsigned int*)facets->get(), (double*)vert->get(),numFacets);
    cuda_sync_and_check("GradV");

}
void CUDA::area_gradient3(UniqueDevicePtr<double>* gradAVert,UniqueDevicePtr<unsigned int>* facets,UniqueDevicePtr<double>* vert,unsigned int numFacets,unsigned int numVert){
	// mem set and call kernal
    cudaMemset(gradAVert->get(),0.0,sizeof(double)*numVert*3);
	cuda_sync_and_check("memset");

	unsigned int numberOfBlocks = ceil(numFacets / (float) blockSize);

	dim3 numberOfBlocks3D( numberOfBlocks, 1, 1 );
	dim3 blockSize3D( blockSize, 3, 1 );

    areaGradient3<<<numberOfBlocks3D, blockSize3D,15*blockSize*sizeof(double)>>>((double*)gradAVert->get(), (unsigned int*)facets->get(), (double*)vert->get(),numFacets);
    cuda_sync_and_check("GradA");
//(double* gradVVert, unsigned int* facets,double* verts,unsigned int numFacets)
}
void CUDA::volume_gradient3(UniqueDevicePtr<double>* gradVVert,UniqueDevicePtr<unsigned int>* facets,UniqueDevicePtr<double>* vert,unsigned int numFacets,unsigned int numVert){
    
	



	// int tmpBlockSize = ceil((float)blockSize/9);
	// // find the closest multiple of 32 
	// tmpBlockSize = ceil((float)tmpBlockSize/32)*32; 


    // unsigned int numberOfBlocks = floor(numFacets / (float) tmpBlockSize);
	// tmpBlockSize = 1;
	// numberOfBlocks = numFacets;
	// dim3 blockSize3D( tmpBlockSize, 3, 3 );
	// unsigned int numberOfBlocks = floor(numFacets / (float) blockSize);
	// dim3 blockSize3D( blockSize, 3, 3 );
	// dim3 numberOfBlocks3D( numberOfBlocks, 1);
	
	cudaMemset(gradVVert->get(),0.0,sizeof(double)*numVert*3);
	cuda_sync_and_check("memset");

	unsigned int tmpBlockSize = 32;
    unsigned int numberOfBlocks = ceil(numFacets / (float) tmpBlockSize);

	dim3 numberOfBlocks3D( numberOfBlocks, 1, 1 );
	dim3 blockSize3D( tmpBlockSize, 3, 3 );
//	printf("calling volGrad3 with %d blocks of size %d X %d X %d \n",numberOfBlocks,blockSize,3,3);

    volumeGradient3<<<numberOfBlocks3D, blockSize3D,blockSize*sizeof(double)>>>((double*)gradVVert->get(), (unsigned int*)facets->get(), (double*)vert->get(),numFacets);
    cuda_sync_and_check("GradV");

}


double CUDA::sum_of_elements(UniqueDevicePtr<double>* vec,unsigned int size,unsigned int bufferedSize){

    double out;
	
    // do the reduction each step sums blockSize*2 number of elements
    unsigned int numberOfBlocks = ceil(size / (float) blockSize / 2.0);
    //printf("AddTree with %d blocks,  of blocks size %d, for %d total elements\n",numberOfBlocks,blockSize,bufferedSize);
    
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
    cudaMemset(scratch->get(),0.0,sizeof(double)*bufferedSize);
	cuda_sync_and_check("memset");

    return out;


}


void CUDA::area(UniqueDevicePtr<double>* area, UniqueDevicePtr<double>* vert, UniqueDevicePtr<unsigned int>* facets, unsigned int numFacets){
    unsigned int numberOfBlocks = ceil(numFacets / (float) blockSize);
    areaKernel<<<numberOfBlocks,blockSize>>>((double*)area->get(),(double*) vert->get(), (unsigned int*) facets->get(), numFacets);
    cuda_sync_and_check("area");

}
void CUDA::volume(UniqueDevicePtr<double>* volume, UniqueDevicePtr<double>* vert, UniqueDevicePtr<unsigned int>* facets, unsigned int numFacets){
	
	cuda_sync_and_check("inb4 volume");
    unsigned int numberOfBlocks = ceil(numFacets / (float) blockSize);
    volumeKernel<<<numberOfBlocks,blockSize>>>((double*)volume->get(),(double*) vert->get(), (unsigned int*) facets->get(), numFacets);
    cuda_sync_and_check("volume");

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
#include "kernalfile.hpp" 

__device__ double atomic_Add(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ void vectorSub(double * v1, double * v2, double * vOut){
    
    *vOut = *v1-*v2;
    *(vOut + 1) = *(v1 + 1) - *(v2 + 1);
    *(vOut + 2) = *(v1 + 2) - *(v2 + 2);
}
__device__ void vectorAdd(double * v1, double * v2, double * vOut) {
    *vOut = *v1 + *v2;
    *(vOut + 1) = *(v1 + 1) + *(v2 + 1);
    *(vOut + 2) = *(v1 + 2) + *(v2 + 2);
}
__device__ void vecScale(double *v, double lambda){
    *v *= lambda;
    *(v+1) *= lambda;
    *(v+2) *= lambda;
}
__device__ void vecAssign(double *out, double *in,double lambda){ // out  = in*lambda
    *out = *in * lambda;
    *(out + 1) = *(in + 1) * lambda;
    *(out + 2) = *(in + 2) * lambda;
}
__device__ void cross(double *a,double *b, double *c) {
    (*c)     = (*(a+1)) * (*(b+2)) - (*(a+2)) * (*(b+1));
    (*(c+1)) = (*(b)) * (*(a+2)) - (*(a)) * (*(b+2));
    (*(c+2)) = (*(a)) * (*(b+1)) - (*(b)) * (*(a+1));
}
__device__ void cross3(double *a,double *b, double *c,int k) {
	if (k == 0){
    	(*c)     = (*(a+1)) * (*(b+2)) - (*(a+2)) * (*(b+1));
	}
	if (k == 1){
    	(*(c+1)) = (*(b)) * (*(a+2)) - (*(a)) * (*(b+2));
	}
	if (k == 2){
    	(*(c+2)) = (*(a)) * (*(b+1)) - (*(b)) * (*(a+1));
	}
}

__device__ double cross3oneComp(double *a,double *b,int k) {
	if (k == 0){
    	//return (*(a+1)) * (*(b+2)) - (*(a+2)) * (*(b+1));
		return a[1]*b[2]-a[2]*b[1];
		
	}
	if (k == 1){
    	//return (*(b)) * (*(a+2)) - (*(a)) * (*(b+2));
		return a[2]*b[0]-a[0]*b[2];
	}
	if (k == 2){
    	//return (*(a)) * (*(b+1)) - (*(b)) * (*(a+1));
		return a[0]*b[1]-a[1]*b[0];
	}
	
	return 0;
}

__device__ double dot(double *a, double *b) {
     return ((*a) * (*b) + (*(a+1)) * (*(b+1)) + (*(a+2)) * (*(b+2)));
}

__device__ double norm(double *a) {
    return sqrt(dot(a, a));
}

__device__ int sign(double a){
    if (a>0) return 1;
    if (a<0) return -1;
    else return 0;
}



__global__ void areaKernel(double * area, double * vert, unsigned int * facets, unsigned int numFacets){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // do i*3 because we have 3 vertcies per facet
    // do facets[]*3 becasue we have x y and z positions
    double r10[3];
    double r21[3];
    double S[3];

    if (i < numFacets) {
        vectorSub(&vert[facets[i*3+1]*3], &vert[facets[i*3]*3],r10);
        vectorSub(&vert[facets[i*3+2]*3], &vert[facets[i*3+1]*3],r21);    
        cross(r10, r21,S);
        area[i] = norm(S)/2;
        //printf("Thread %d:\tArea %f\n",i,area[i]);
    }
    else {
        area[i] = 0;
    }
}
__global__ void volumeKernel(double * volume, double * vert, unsigned int * facets, unsigned int numFacets){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    double s01[3];
    if (i < numFacets){
        cross(&vert[facets[i*3]*3], &vert[facets[i*3+1]*3],s01);
        volume[i] = abs(dot(s01,&vert[facets[i*3+2]*3]))/6;
    }
    else {
        volume[i] = 0;
    }

}
__global__ void addTree(double* g_idata, double* g_odata){
    //https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

    extern __shared__ double sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x; // get the id of this thread
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
        //printf("tid: %d\ti:%d\ti + blockDim.x:%d\tg_idata[i]:%f\tg_idata[i + blockDim.x]%f\n",tid,i, i + blockDim.x, g_idata[i] , g_idata[i + blockDim.x]);
        sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];// g_idata[i]; // move the data over
        g_idata[i] = 0;
        g_idata[i + blockDim.x] = 0;
        // printf("tid: %d\ti:%d\ti + blockDim.x:%d\tg_idata[i]:%f\tg_idata[i + blockDim.x]%f\t sdata[tid]: %f\n", tid, i, i + blockDim.x, g_idata[i], g_idata[i + blockDim.x], sdata[tid]);
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

// template <unsigned int blockSize> __device__ void warpReduce(volatile double *sdata, unsigned int tid) {
//         if (blockSize >=  64) sdata[tid] += sdata[tid + 32];
//         if (blockSize >=  32) sdata[tid] += sdata[tid + 16];
//         if (blockSize >=  16) sdata[tid] += sdata[tid +  8];
//         if (blockSize >=    8) sdata[tid] += sdata[tid +  4];
//         if (blockSize >=    4) sdata[tid] += sdata[tid +  2];
//         if (blockSize >=    2) sdata[tid] += sdata[tid +  1];
// }
// template <unsigned int blockSize> __global__ void reduce6(double *g_idata,double *g_odata, unsigned int n) {
//     extern __shared__ double sdata[];
//     unsigned int tid = threadIdx.x;
//     unsigned int i = blockIdx.x*(blockSize*2) + tid;
//     unsigned int gridSize = blockSize*2*gridDim.x;
//     sdata[tid] = 0;
//     while (i < n) {
//         sdata[tid] += g_idata[i] + g_idata[i+blockSize];
//         i += gridSize;
//     }
//     __syncthreads();
//     if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
//     if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
//     if (blockSize >= 128) { if (tid <   64) { sdata[tid] += sdata[tid +   64]; } __syncthreads(); }
//     if (tid < 32) warpReduce(sdata, tid);
//     if (tid == 0) g_odata[blockIdx.x] = sdata[0];
// }

__global__ void addWithMultKernel(double *a ,double *b,double lambda, unsigned int size){
    // a += b * lambda
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<size){
        *(a+i) += *(b+i) * lambda;
    }
}

__global__ void areaGradient(double* gradAFacet, unsigned int* facets,double* verts,unsigned int numFacets){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    double S0[3];
    double S1[3];
    double S01[3];
    double S010[3];
    double S011[3];
    if (i<numFacets){
        vectorSub(&verts[facets[i*3+1]*3], &verts[facets[i*3]*3],S0);
        vectorSub(&verts[facets[i*3+2]*3], &verts[facets[i*3+1]*3],S1);
        cross(S0,S1,S01);
        cross(S01,S0,S010);
        cross(S01,S1,S011);
        // each facet has 3 vertices with gradient each, so in total 9 numbers we write them down here;
        
        // or facet i this is the gradent vector for its 0th vertex 
        vecAssign(&gradAFacet[i*9],S011,1.0/(2 * norm(S01)));

        // reuse S0 
        vectorAdd(S011,S010,S0);
        vecAssign(&gradAFacet[i*9 + 3],S0,-1.0/(2 * norm(S01)));

        vecAssign(&gradAFacet[i*9 + 6],S010,1.0/(2 * norm(S01)));
    }

}

__global__ void areaGradient2(double* gradAVert, unsigned int* facets,double* verts, unsigned int numFacets){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    double S0[3];
    double S1[3];
    double S01[3];
    double S010[3];
    double S011[3];

    if (i<numFacets){
        vectorSub(&verts[facets[i*3+1]*3], &verts[facets[i*3]*3],S0);
        vectorSub(&verts[facets[i*3+2]*3], &verts[facets[i*3+1]*3],S1);
        cross(S0,S1,S01);
        cross(S01,S0,S010);
        cross(S01,S1,S011);
        // each facet has 3 vertices with gradient each, so in total 9 numbers we write them down here;
        
        // or facet i this is the gradent vector for its 0th vertex 
        vectorAdd(S011,S010,S0);
		double Norm2inv = 1.0/(2.0*norm(S01));
		for (int j=0; j<3;j++){
			atomic_Add(&gradAVert[facets[i*3]*3+j],S011[j]*Norm2inv);
			atomic_Add(&gradAVert[facets[i*3+1]*3+j],-S0[j]*Norm2inv);
			atomic_Add(&gradAVert[facets[i*3+2]*3+j],S010[j]*Norm2inv);
		}



				// 		gradAVert[facets[i*3+1]*3]  -=S0[0]*Norm2inv;
		// 		gradAVert[facets[i*3+1]*3+1]-=S0[1]*Norm2inv;
		// 		gradAVert[facets[i*3+1]*3+2]-=S0[2]*Norm2inv;

						// 		gradAVert[facets[i*3+2]*3]  +=S010[0]*Norm2inv;
		// 		gradAVert[facets[i*3+2]*3+1]+=S010[1]*Norm2inv;
		// 		gradAVert[facets[i*3+2]*3+2]+=S010[2]*Norm2inv;



		// for (int ii = 0; ii<=maxAdd; ii++){
		// 	 __syncthreads(); // problem is that this is a local sync got a global one, we really should be doing a global one here
		// 	if (addOrder[i*3]==ii){
				
		// 		gradAVert[facets[i*3]*3]  +=S011[0]*Norm2inv;
		// 		gradAVert[facets[i*3]*3+1]+=S011[1]*Norm2inv;
		// 		gradAVert[facets[i*3]*3+2]+=S011[2]*Norm2inv;
		// 		if (facets[i*3]==VERTTOCHECK){
		// 			printf("adding {%f,%f,%f} from facet %d i am %d in the add order\t",S011[0]*Norm2inv,S011[1]*Norm2inv,S011[2]*Norm2inv,i,addOrder[i*3]);
		// 			printf("got {%f,%f,%f}\n",gradAVert[0+3*VERTTOCHECK],gradAVert[1+3*VERTTOCHECK],gradAVert[2+3*VERTTOCHECK]);
		// 		}
				
		// 	}
		// 	if (addOrder[i*3+1]==ii){
		// 		gradAVert[facets[i*3+1]*3]  -=S0[0]*Norm2inv;
		// 		gradAVert[facets[i*3+1]*3+1]-=S0[1]*Norm2inv;
		// 		gradAVert[facets[i*3+1]*3+2]-=S0[2]*Norm2inv;
		// 		if (facets[i*3+1]==VERTTOCHECK){
		// 			printf("adding {%f,%f,%f} from facet %d i am %d in the add order\t",S0[0]*Norm2inv,S0[1]*Norm2inv,S0[2]*Norm2inv,i,addOrder[i*3+2]);
		// 			printf("got {%f,%f,%f}\n",gradAVert[0+3*VERTTOCHECK],gradAVert[1+3*VERTTOCHECK],gradAVert[2+3*VERTTOCHECK]);
		// 		}
		// 	}
		// 	if (addOrder[i*3+2]==ii){
		// 		gradAVert[facets[i*3+2]*3]  +=S010[0]*Norm2inv;
		// 		gradAVert[facets[i*3+2]*3+1]+=S010[1]*Norm2inv;
		// 		gradAVert[facets[i*3+2]*3+2]+=S010[2]*Norm2inv;
		// 		if (facets[i*3+2]==VERTTOCHECK){
		// 			printf("adding {%f,%f,%f} from facet %d i am %d in the add order\t",S010[0]*Norm2inv,S010[1]*Norm2inv,S010[2]*Norm2inv,i,addOrder[i*3+2]);
		// 			printf("got {%f,%f,%f}\n",gradAVert[0+3*VERTTOCHECK],gradAVert[1+3*VERTTOCHECK],gradAVert[2+3*VERTTOCHECK]);
		// 		}
		// 	}
		// }
    }

}
__global__ void areaGradient3(double* gradAVert, unsigned int* facets,double* verts, unsigned int numFacets){
    int i = blockDim.x * blockIdx.x + threadIdx.x; //which facet is calculating
	//int j = threadIdx.y; // which vertex on this facet are we calculating for
	int k = threadIdx.y; // which (x,y,z) compoent of the vertex are we calculating
    
	extern __shared__ double sdata[]; // need 5x3xblocksize double for the whole block; I.E each thread in the block needs 5 3 vectors
    double *S0 = sdata+5*3*threadIdx.x;
    double *S1 = S0+3;
    double *S01 = S1+3;
    double *S010 = S01+3;
    double *S011 = S010+3;

    if (i<numFacets){
		S0[k] = verts[facets[i*3+1]*3+k] - verts[facets[i*3]*3+k];
		S1[k] = verts[facets[i*3+2]*3+k] - verts[facets[i*3+1]*3+k];
		__syncthreads();

		cross3(S0,S1,S01,k);
		__syncthreads();
        
		cross3(S01,S0,S010,k);
        cross3(S01,S1,S011,k);
		__syncthreads();
        S0[k] = S011[k]+S010[k];
        __syncthreads();
		// hmmm this is pretty in effiecnt 
        // or facet i this is the gradent vector for its 0th vertex 
		double Norm2inv = 1.0/(2.0*norm(S01));
		atomic_Add(&gradAVert[facets[i*3]*3+k],S011[k]*Norm2inv);
		atomic_Add(&gradAVert[facets[i*3+1]*3+k],-S0[k]*Norm2inv);
		atomic_Add(&gradAVert[facets[i*3+2]*3+k],S010[k]*Norm2inv);
	}
}

__global__ void volumeGradient(double* gradVFacet, unsigned int* facets,double* verts,unsigned int numFacets){
    // TO DO: this can this can be broken up into 3 for even faster computaiton
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    double c[3];
    double s = 1;
    if (i<numFacets){
        cross(&verts[facets[i*3]*3],&verts[facets[i*3+1]*3],c);
        s = sign(dot(c,&verts[facets[i*3+2]*3]));

        cross(&verts[facets[i*3+1]*3],&verts[facets[i*3+2]*3],c);
        vecAssign(&gradVFacet[i*9],c,s/6);

        cross(&verts[facets[i*3+2]*3],&verts[facets[i*3]*3],c);
        vecAssign(&gradVFacet[i*9 + 3],c,s/6);

        cross(&verts[facets[i*3]*3],&verts[facets[i*3+1]*3],c);
        vecAssign(&gradVFacet[i*9 + 6],c,s/6);
    }

}

__global__ void volumeGradient2(double* gradVVert, unsigned int* facets,double* verts,unsigned int numFacets){
    // TO DO: this can this can be broken up into 3 for even faster computaiton
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    double a[3];
	double b[3];
	double c[3];
    double s = 1;
    if (i<numFacets){
        cross(&verts[facets[i*3]*3],&verts[facets[i*3+1]*3],c);
        s = sign(dot(c,&verts[facets[i*3+2]*3]));
		//printf("facet %d, s = %f, c= {%f,%f,%f}\n",i,s,c[0],c[1],c[2]);
        cross(&verts[facets[i*3+1]*3],&verts[facets[i*3+2]*3],a);
        cross(&verts[facets[i*3+2]*3],&verts[facets[i*3]*3],b);
        cross(&verts[facets[i*3]*3],&verts[facets[i*3+1]*3],c);

		for (int j=0; j<3;j++){
			atomic_Add(&gradVVert[facets[i*3]*3+j],a[j]*s/6);
			// if (a[j]*s/6>1e-6){
			// printf("I am Facet %d, place %d, component %d and my term is %f I have added to vertex %d it is now %f\n",i,0,j,a[j]*s/6,facets[i*3],gradVVert[facets[i*3]*3+j]);
			// }

			atomic_Add(&gradVVert[facets[i*3+1]*3+j],b[j]*s/6);
			// if (b[j]*s/6>1e-6){
			// printf("I am Facet %d, place %d, component %d and my term is %f I have added to vertex %d it is now %f\n",i,1,j,b[j]*s/6,facets[i*3+1],gradVVert[facets[i*3+1]*3+j]);
			// }

			atomic_Add(&gradVVert[facets[i*3+2]*3+j],c[j]*s/6);
			// if (c[j]*s/6>1e-6){
			// printf("I am Facet %d, place %d, component %d and my term is %f I have added to vertex %d it is now %f\n",i,2,j,c[j]*s/6,facets[i*3+2],gradVVert[facets[i*3+2]*3+j]);
			// }

		}
    }

}
__global__ void volumeGradient3(double* gradVVert, unsigned int* facets,double* verts,unsigned int numFacets){
    // TO DO: this can this can be broken up into 3 for even faster computaiton
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = threadIdx.y; // which vertex on this facet are we calculating for
	int k = threadIdx.z; // which (x,y,z) compoent of the vertex are we calculating
	extern __shared__ double sdata[]; // just one piece of data per facet
	// lets load the 3 vectors into shared data frist so now we need 10 data per facet
	if (j ==0 && k ==0) sdata[threadIdx.x] = 0;

	__syncthreads();


	double term; 
    
	double s = 1;



    if (i<numFacets){
		if (j==0){
        	term = cross3oneComp(&verts[facets[i*3]*3],&verts[facets[i*3+1]*3],k);
			// now take the sign of c dotted with &verts[facets[i*3+2]*3]
			atomic_Add(&sdata[threadIdx.x],term*verts[facets[i*3+2]*3+k]);
		}
	}
	__syncthreads();

    if (i<numFacets){

        s = sign(sdata[threadIdx.x]);
		term = cross3oneComp(&verts[facets[i*3+(j+1)%3]*3],&verts[facets[i*3+(j+2)%3]*3],k);

		atomic_Add(&gradVVert[facets[i*3+j]*3+k],term*s/6.0);
	}
}
		

__global__ void facetToVertex(double* vertexValue, double* facetValue,unsigned int* vertToFacet, unsigned int* vertIndexStart,unsigned int numVert){
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = threadIdx.y;

    if (i<numVert){
        //first set to 0
        vertexValue[i*3 + j] = 0;
        for (int index = vertIndexStart[i]; index < vertIndexStart[i+1]; index++){
			vertexValue[i*3+j]+=facetValue[3*vertToFacet[index]+j];
            //vectorAdd(&vertexValue[i*3],&facetValue[3*vertToFacet[index]],&vertexValue[i*3]);
            //printf("vertex %d gets [%f,%f,%f]\n",i,facetValue[3*vertToFacet[index]],facetValue[3*vertToFacet[index]+1],facetValue[3*vertToFacet[index]+2]);
        }
    }
}

__global__ void projectForce(double* force,double* gradAVert,double* gradVVert,double scale,unsigned int numEle){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<numEle){
        force[i] = - (gradAVert[i] - scale * gradVVert[i]);
    }
}

__global__ void elementMultiply(double* v1, double* v2, double* out, unsigned int size){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<size){
        out[i] = v1[i]*v2[i];
        //printf("Thread %d: out value %f\n",i,out[i]);
    }
}


