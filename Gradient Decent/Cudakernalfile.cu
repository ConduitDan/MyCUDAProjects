#include "kernalfile.hpp"

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
__global__ void facetToVertex(double* vertexValue, double* facetValue,unsigned int* vertToFacet, unsigned int* vertIndexStart,unsigned int numVert){
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i<numVert){
        //first set to 0
        vertexValue[i*3] = 0;
        vertexValue[i*3 + 1] = 0;
        vertexValue[i*3 + 2] = 0;
        for (int index = vertIndexStart[i]; index < vertIndexStart[i+1]; index++){
            vectorAdd(&vertexValue[i*3],&facetValue[3*vertToFacet[index]],&vertexValue[i*3]);
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


