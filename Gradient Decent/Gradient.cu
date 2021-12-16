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
    // the force vector is used as scrach for taking the dot products for projection,
    // so it needs to be padded to a multiple of twice the block size so we can effiecntly sum it
    unsigned int bufferedSize = ceil(numVert * 3 /(2.0*_myMesh->get_blockSize()))*2*_myMesh->get_blockSize();

    
    _cudaStatus = cudaMalloc((void**)&_scratch, bufferedSize * sizeof(double));
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }


}

Gradient::~Gradient(){
    if (_gradAFacet){
	    cudaFree(_gradAFacet);
	    _gradAFacet = nullptr;
    }
    if (_gradAVert){
	    cudaFree(_gradAVert);
    	_gradAVert = nullptr;
    }
    if (_gradVFacet){
	    cudaFree(_gradVFacet);
	    _gradVFacet = nullptr;
    }
    if (_gradVVert){
	    cudaFree(_gradVVert);
	    _gradVVert = nullptr;
    }
    if (_force){
	    cudaFree(_force);
	    _force = nullptr;
    }
    if (_scratch){
	    cudaFree(_scratch);
	_scratch = nullptr;
    }
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
    cuda_sync_and_check(_cudaStatus,"area gradient");

    facet_to_vertex(_gradAFacet,_gradAVert);

}

void Gradient::calc_gradV(){
    // first calculate the gradient on the facets
    unsigned int numberOfBlocks = ceil(_myMesh->get_numFacets() / (float) _myMesh->get_blockSize());
    volumeGradient<<<numberOfBlocks,_myMesh->get_blockSize()>>>(_gradVFacet,_myMesh->get_facets(),_myMesh->get_vert(),_myMesh->get_numFacets());
    cuda_sync_and_check(_cudaStatus,"volume gradient");

    facet_to_vertex(_gradVFacet,_gradVVert);

}

void Gradient::facet_to_vertex(double* _facetValue,double* _vertexValue){

    unsigned int numberOfBlocks = ceil(_myMesh->get_numVert() / (float) _myMesh->get_blockSize());
    facetToVertex<<<numberOfBlocks,_myMesh->get_blockSize()>>>(_vertexValue,_facetValue,_myMesh->get_vertToFacet(), _myMesh->get_vertIndexStart(),_myMesh->get_numVert());
    cuda_sync_and_check(_cudaStatus,"face to vertex");

}

void Gradient::project_to_force(){
    double numerator = dotProduct(_cudaStatus,_gradAVert,_gradVVert,_scratch,_myMesh->get_numVert() * 3,_myMesh->get_blockSize() );
    double denominator = dotProduct(_cudaStatus,_gradVVert,_gradVVert,_scratch,_myMesh->get_numVert() * 3,_myMesh->get_blockSize() );

    unsigned int numberOfBlocks = ceil(_myMesh->get_numVert() * 3 / (float) _myMesh->get_blockSize());

    projectForce<<<numberOfBlocks,_myMesh->get_blockSize()>>>(_force,_gradAVert,_gradVVert,numerator/abs(denominator),_myMesh->get_numVert() * 3);
    cuda_sync_and_check(_cudaStatus,"project force");

}

void Gradient::reproject(double res){
    calc_gradV();
    // do the force inner product
    double sol = res/dotProduct(_cudaStatus,_gradVVert,_gradVVert,_scratch,_myMesh->get_numVert(),_myMesh->get_blockSize());
    
    //move and scale (scale = sol, dir = gradV)
    unsigned int numberOfBlocks = ceil(_myMesh->get_numVert() * 3 / (float) _myMesh->get_blockSize());

    addWithMultKernel<<<numberOfBlocks,_myMesh->get_blockSize()>>>(_myMesh->get_vert(),_gradVVert,sol*0.1,_myMesh->get_numVert()*3);

}
