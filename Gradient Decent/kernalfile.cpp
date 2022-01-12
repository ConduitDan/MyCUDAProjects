//OpenCLKernals.cl
//#ifndef KERNALS
//#define KERNALS

#include "kernalfile.hpp"



DPREFACTOR void vectorSub(double * v1, double * v2, double * vOut){
    
    *vOut = *v1-*v2;
    *(vOut + 1) = *(v1 + 1) - *(v2 + 1);
    *(vOut + 2) = *(v1 + 2) - *(v2 + 2);
}
DPREFACTOR void vectorAdd(double * v1, double * v2, double * vOut) {
    *vOut = *v1 + *v2;
    *(vOut + 1) = *(v1 + 1) + *(v2 + 1);
    *(vOut + 2) = *(v1 + 2) + *(v2 + 2);
}
DPREFACTOR void vecScale(double *v, double lambda){
    *v *= lambda;
    *(v+1) *= lambda;
    *(v+2) *= lambda;
}
DPREFACTOR void vecAssign(double *out, double *in,double lambda){ // out  = in*lambda
    *out = *in * lambda;
    *(out + 1) = *(in + 1) * lambda;
    *(out + 2) = *(in + 2) * lambda;
}
DPREFACTOR void cross(double *a,double *b, double *c) {
    (*c)     = (*(a+1)) * (*(b+2)) - (*(a+2)) * (*(b+1));
    (*(c+1)) = (*(b)) * (*(a+2)) - (*(a)) * (*(b+2));
    (*(c+2)) = (*(a)) * (*(b+1)) - (*(b)) * (*(a+1));
}

DPREFACTOR double dot(double *a, double *b) {
     return ((*a) * (*b) + (*(a+1)) * (*(b+1)) + (*(a+2)) * (*(b+2)));
}

DPREFACTOR double norm(double *a) {
    return sqrt(dot(a, a));
}

DPREFACTOR int sign(double a){
    if (a>0) return 1;
    if (a<0) return -1;
    else return 0;
}



PREAMBLE(areaKernel) PREP_FOR_PARSE(void areaKernel(DEVTAG double * area,DEVTAG double * vert,DEVTAG unsigned int * facets, unsigned int numFacets){
    int i = GETID;
    double r10[3];
    double r21[3];
    double S[3];

    if (i < numFacets) {
        vectorSub(&vert[facets[i*3+1]*3], &vert[facets[i*3]*3],r10);
        vectorSub(&vert[facets[i*3+2]*3], &vert[facets[i*3+1]*3],r21);    
        cross(r10, r21,S);
        area[i] = norm(S)/2;
    }
    else {
        area[i] = 0;
    }
})




PREAMBLE(volumeKernel) PREP_FOR_PARSE( void volumeKernel(DEVTAG double * volume, DEVTAG double * vert, DEVTAG unsigned int * facets, unsigned int numFacets){
    int i = GETID;

    double s01[3];
    if (i < numFacets){
        cross(&vert[facets[i*3]*3], &vert[facets[i*3+1]*3],s01);
        volume[i] = abs(dot(s01,&vert[facets[i*3+2]*3]))/6;
    }
    else {
        volume[i] = 0;
    }

})
PREAMBLE(addTree) PREP_FOR_PARSE( void addTree(DEVTAG double* g_idata, DEVTAG double* g_odata OPENCLSHARED){

    CUDASHARED
    unsigned int tid = WORKITEMID; 

    unsigned int i = BLOCKID * 2 + WORKITEMID;
    
        sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
        g_idata[i] = 0;
        g_idata[i + blockDim.x] = 0;
    __syncthreads();
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
        __syncthreads();
        }
        __syncthreads();
        
        if (tid == 0) g_odata[blockIdx.x] = sdata[0];
})



PREAMBLE(addWithMultKernel) PREP_FOR_PARSE( void addWithMultKernel(DEVTAG double *a ,DEVTAG double *b, double lambda, unsigned int size){
    int i = GETID;
    if (i<size){
        *(a+i) += *(b+i) * lambda;
    }
})

PREAMBLE(areaGradient) PREP_FOR_PARSE( void areaGradient( DEVTAG double* gradAFacet, DEVTAG unsigned int* facets,DEVTAG double* verts,unsigned int numFacets){
    int i = GETID;

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

        vecAssign(&gradAFacet[i*9],S011,1.0/(2 * norm(S01)));

        vectorAdd(S011,S010,S0);
        vecAssign(&gradAFacet[i*9 + 3],S0,-1.0/(2 * norm(S01)));

        vecAssign(&gradAFacet[i*9 + 6],S010,1.0/(2 * norm(S01)));
    }

})
PREAMBLE(volumeGradient) PREP_FOR_PARSE( void volumeGradient(DEVTAG double* gradVFacet,DEVTAG unsigned int* facets, DEVTAG double* verts,unsigned int numFacets){
    int i = GETID;
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

})
PREAMBLE(facetToVertex) PREP_FOR_PARSE( void facetToVertex(DEVTAG double* vertexValue,DEVTAG double* facetValue, DEVTAG unsigned int* vertToFacet, DEVTAG unsigned int* vertIndexStart,unsigned int numVert){
    
    int i = GETID;

    if (i<numVert){
        vertexValue[i*3] = 0;
        vertexValue[i*3 + 1] = 0;
        vertexValue[i*3 + 2] = 0;
        for (int index = vertIndexStart[i]; index < vertIndexStart[i+1]; index++){
            vectorAdd(&vertexValue[i*3],&facetValue[3*vertToFacet[index]],&vertexValue[i*3]);
        }
    }
})

PREAMBLE(projectForce) PREP_FOR_PARSE( void projectForce( DEVTAG double* force, DEVTAG double* gradAVert,DEVTAG double* gradVVert,double scale,unsigned int numEle){
    int i =GETID;
    if (i<numEle){
        force[i] = - (gradAVert[i] - scale * gradVVert[i]);
    }
})

PREAMBLE(elementMultiply) PREP_FOR_PARSE( void elementMultiply(DEVTAG double* v1, DEVTAG double* v2, DEVTAG double* out, unsigned int size){
    int i = GETID;
    if (i<size){
        out[i] = v1[i]*v2[i];
    }
})






//#endif