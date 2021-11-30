#include "Gradient.hpp"


Gradient::Gradient(DeviceMesh *inMesh){
    _myMesh = inMesh;
    unsigned int numVert = _myMesh->get_numVert();
    unsigned int numFacet = _myMesh->get_numFacets();

    _cudaStatus = cudaMalloc((void**)&_gradAFacet, numFacet * 3 * 3 * sizeof(double));
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
        _cudaStatus = cudaMalloc((void**)&_gradAVert, numVert * 3 * sizeof(double));
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

        _cudaStatus = cudaMalloc((void**)&_gradVFacet, numFacet * 3 * 3 * sizeof(double));
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
        _cudaStatus = cudaMalloc((void**)&_gradVVert, numVert * 3 * sizeof(double));
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }


        _cudaStatus = cudaMalloc((void**)&_force, numVert * 3 * sizeof(double));
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
}

Gradient::~Gradient(){
    if (_gradAFacet) cudaFree(_gradAFacet);
    if (_gradAVert) cudaFree(_gradAVert);
    if (_gradVFacet) cudaFree(_gradVFacet);
    if (_gradVVert) cudaFree(_gradVVert);
    if (_force) cudaFree(_force);

}
    
    
void Gradient::calc_force(){
    calc_gradA();
	calc_gradV();
	project_to_force();
}

void Gradient::calc_gradA(){
    // first calculate the gradient on the facets
    unsigned int numberOfBlocks = ceil(_myMesh->get_numFacets() / (float) _myMesh->get_blockSize());
    areaGradient<<<numberOfBlocks,_myMesh->get_blockSize()>>>(_gradAFacet,_myMesh->get_facets(),_myMesh->get_vert(),_myMesh->get_numFacets());
    cuda_sync_and_check("area gradient");

    facet_to_vertex(_gradAFacet,_gradAVert);

}

void Gradient::calc_gradV(){
    // first calculate the gradient on the facets
    unsigned int numberOfBlocks = ceil(_myMesh->get_numFacets() / (float) _myMesh->get_blockSize());
    volumeGradient<<<numberOfBlocks,_myMesh->get_blockSize()>>>(_gradVFacet,_myMesh->get_facets(),_myMesh->get_vert(),_myMesh->get_numFacets());
    cuda_sync_and_check("volume gradient");

    facet_to_vertex(_gradVFacet,_gradVVert);

}

void Gradient::facet_to_vertex(double* _facetValue,double* _vertexValue){

    unsigned int numberOfBlocks = ceil(_myMesh->get_numVert() / (float) _myMesh->get_blockSize());
    facetToVertex<<<numberOfBlocks,_myMesh->get_blockSize()>>>(_vertexValue,_facetValue,_myMesh->get_vertToFacet(), _myMesh->get_vertIndexStart(),_myMesh->get_numVert());
    cuda_sync_and_check("face to vertex");

}
void Gradient::project_to_force(){

    unsigned int numberOfBlocks = ceil(_myMesh->get_numVert() / (float) _myMesh->get_blockSize());
    projectForce<<<numberOfBlocks,_myMesh->get_blockSize()>>>(_force,_gradAVert,_gradVVert,_myMesh->get_numVert());
    cuda_sync_and_check("project force");

}

void Gradient::cuda_sync_and_check(const char* caller){
    _cudaStatus = cudaGetLastError();
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s from %s\n", cudaGetErrorString(_cudaStatus),caller);
        throw;
    }
    // check that the kernal didn't throw an error
    _cudaStatus = cudaDeviceSynchronize();
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error %s after launching Kernel %s!\n", cudaGetErrorString(_cudaStatus),caller);
        throw;
    }

}