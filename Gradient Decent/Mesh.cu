#include "Mesh.hpp"
#include "Kernals.cu"


Mesh::Mesh(const char* fileName){
    if(load_mesh_from_file(fileName)){
        fprintf(stdout,"Successfully read %s \n",fileName);
    }
    else {
        fprintf(stdout,"Failed to read %s \n",fileName);
    }

}

Mesh::~Mesh(){

    if (_vert) delete _vert;
    if (_facets) delete _facets;
}



int Mesh::getNumVertices(FILE* fp) {

    int numVertices = 0;
    char line[50];
    fscanf(fp, "%s\n", line);
    if (strcmp(line, "vertices")) {
        fprintf(stderr, "File didn't start with 'vertices'\n");
        return -1;

    }
    fgets(line, 50, fp); // eat the new line
    fgets(line, 50, fp); // read line 1
    numVertices++;
    while (strcmp(line, "\n") && !feof(fp)) {
        numVertices++;
        fgets(line, 50, fp); // read line n
    }
    return numVertices;

}



int Mesh::getNumFacets(FILE* fp) {
    int numFaces = 0;
    char line[50];
    fscanf(fp, "%s\n", line);
    while (strcmp(line, "faces")) {
        fscanf(fp, "%s\n", line);
        if (feof(fp)) {
            fprintf(stderr, "File had no faces\n");
            return -1;
        }
    }
    fgets(line, 50, fp); // eat the new line
    fgets(line, 50, fp); // read line 1
    numFaces++;
    while (strcmp(line, "\n") && !feof(fp)) {
        numFaces++;
        fgets(line, 50, fp); // read line 1

    }
    return numFaces;

}



bool Mesh::load_mesh_from_file(const char* fileName) {
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

    _numVert = getNumVertices(fp);
    _numFacets = getNumFacets(fp);

    _vert = new double[_numVert * 3]; // [x0; y0; z0; x1; y1;.... ]
    _facets =  new unsigned int[_numFacets * 3];;// [a0; b0; c0; a1;b1;c1;...]


    rewind(fp); // rewind the file to the beginning
    // make sure the first line say vertices

    fscanf(fp, "%s\n", sectionHeader);
    if (strcmp(sectionHeader, "vertices")) {
        fprintf(stderr, "File didn't start with 'vertices'\n");
        return false;
    }
    // get past the empty line

    float tmp0, tmp1, tmp2;
    for (int i = 0; i < _numVert; i++) {
        numAssigned = fscanf(fp, "%*d %f %f %f\n", &tmp0, &tmp1, &tmp2);

        if (numAssigned < 3) {
            fprintf(stderr, "bad file format\n");
            return false;
        }
        _vert[i*3] = tmp0;
        _vert[i*3 + 1] = tmp1;
        _vert[i*3 + 2] = tmp2;
    }

    fscanf(fp, "%*d");
    fscanf(fp, "%s\n", sectionHeader);
    while (strcmp(sectionHeader, "faces")) {
        fscanf(fp, "%s\n", sectionHeader);
        if (feof(fp)) {
            fprintf(stderr, "couldn't find faces\n");
            return false;
        }
    }

    for (int i = 0; i < _numFacets; i++) {
        numAssigned = fscanf(fp, "%*d %d %d %d\n", _facets + i * 3, _facets + i * 3 + 1, _facets + i * 3 + 2);
        if (numAssigned < 3) {
            fprintf(stderr, "bad file format for faces\n");
            return false;
        }
        _facets[i * 3] --;
        _facets[i * 3 + 1] --;
        _facets[i * 3 + 2] --;

    }
    return true;
}



DeviceMesh::DeviceMesh(Mesh hostMesh, unsigned int blockSize){
    // set up the GPU 
    _cudaStatus = cudaSetDevice(0);
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    
    
    // copy over the number of elements
    _numFacets = hostMesh.get_numFacets();
    _numVert = hostMesh.get_numVert();

    // create the vertex-><facets, # in facet> map
    // there should be 3*_numfacets values in the map.
    // we should store a compaion array with the starting index for vertex i in the map

    unsigned int *vertToFacet  = new unsigned int [_numFacets * 3] {0};// (store i*3 + j);
    //unsigned int *vertToFacetIndex = new unsigned int [_numFacets * 3] {0};
    unsigned int *vertIndexStart = new unsigned int [_numVert+1] {0}; // + one more here so we have[0...numVert] to make logic simpler for first or last ele
    unsigned int *vertCount = new unsigned int [_numVert] {0};

    // TODO: memset here
    
    unsigned int* hostFacets = hostMesh.get_facets();

    // first fill out how many facets each vertex participates in; 
    for (int i = 0; i<_numFacets * 3; i++){
        vertIndexStart[*(hostFacets+i)+1]++;
    }
    // add the previous entry so it now marks where the entries end for this vertex
    for (int i = 0; i<_numVert; i++){
        vertIndexStart[i+1]+=vertIndexStart[i];
    }
    unsigned int index;
    unsigned int vertex; 
    for (int i = 0; i<_numFacets*3; i++){

        vertex = *(hostFacets + i);
        index = vertIndexStart[vertex]+vertCount[vertex];

        vertToFacet[index] = i;
        vertCount[vertex]++;
    }

    





    _blockSize = blockSize;


    //unsigned int BufferedSize = ceil(_numFacets / (float)(2 * blockSize)) * 2 * blockSize; // for 

    // Allocate GPU buffers for vertices and facets 

    _cudaStatus = cudaMalloc((void**)&_vert, _numVert * 3 * sizeof(double));
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    _cudaStatus = cudaMalloc((void**)&_facets, _numFacets * 3 * sizeof(unsigned int));
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    // and the map for vertex-> facet
    _cudaStatus = cudaMalloc((void**)&_vertToFacet, _numFacets * 3 * sizeof(unsigned int));
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    _cudaStatus = cudaMalloc((void**)&_vertIndexStart, _numVert * sizeof(unsigned int));
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    // and the area and volume 
    _cudaStatus = cudaMalloc((void**)&_area, _numFacets * sizeof(double));
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    _cudaStatus = cudaMalloc((void**)&_volume, _numFacets * sizeof(double));
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    


    // copy over the vertices and facets
    _cudaStatus = cudaMemcpy(_vert, hostMesh.get_vert(), _numVert * 3 * sizeof(double), cudaMemcpyHostToDevice);
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! vertices\n");
    }
    _cudaStatus = cudaMemcpy(_facets, hostMesh.get_facets(), _numFacets * 3 * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! vertices\n");
    }
    // and the map for vertex-> facet
        _cudaStatus = cudaMemcpy(_vertToFacet, vertToFacet, _numFacets * 3 * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! vertices\n");
    }
    _cudaStatus = cudaMemcpy(_vertIndexStart, vertIndexStart, _numVert * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! vertices\n");
    }





    delete[] vertToFacet;
    delete[] vertIndexStart;
    delete[] vertCount;

}

DeviceMesh::~DeviceMesh(){
    if (_vert) cudaFree(_vert);
    if (_facets) cudaFree(_facets);
    if (_vertToFacet) cudaFree(_facets);
    if (_vertIndexStart) cudaFree(_facets);
    if (_area) cudaFree(_area);
    //if (_areaSum) cudaFree(_areaSum);
    if (_volume) cudaFree(_volume);
    //if (_volumeSum) cudaFree(_volumeSum);
}

Mesh DeviceMesh::copy_to_host(){
    Mesh newMesh;
    newMesh._numVert = _numVert;
    newMesh._numFacets = _numFacets;
    newMesh._vert = new double[_numVert * 3]; // [x0; y0; z0; x1; y1;.... ]
    newMesh._facets =  new unsigned int[_numFacets * 3];;// [a0; b0; c0; a1;b1;c1;...]

    _cudaStatus = cudaMemcpy(newMesh._vert, _vert, _numVert * 3 *  sizeof(double), cudaMemcpyDeviceToHost);
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
    }

    _cudaStatus = cudaMemcpy(newMesh._facets, _facets, _numFacets * 3 *  sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
    }


    return newMesh;
}

double DeviceMesh::area(){

    unsigned int numberOfBlocks = ceil(_numFacets / (float) _blockSize);
    areaKernel<<<numberOfBlocks, _blockSize>>> (_area, _vert, _facets, _numFacets);
    cuda_sync_and_check();
    return sum_of_elements(_area,_numFacets);

}

double DeviceMesh::volume(){
    unsigned int numberOfBlocks = ceil(_numFacets / (float) _blockSize);

    volumeKernel<<<numberOfBlocks, _blockSize>>> (_volume, _vert, _facets, _numFacets);

    cuda_sync_and_check();
    return sum_of_elements(_volume,_numFacets);

}

double DeviceMesh::sum_of_elements(double* _vec,unsigned int size){

    double* out;

    // do the reduction each step sums _blockSize*2 number of elements
    unsigned int numberOfBlocks = ceil(size / (float) _blockSize / 2.0);
    addTree<<<numberOfBlocks, BLOCKSIZE, BufferedSize / 2 * sizeof(double) >>> (_area, _area);

    for (int i = numberOfBlocks; i > 1; i /= (_blockSize * 2)) {
      addTree<<<ceil((float)numberOfBlocks/ (_blockSize * 2)), _blockSize, ceil((float)size / 2)* sizeof(double) >>> (_vec, _vec);
    } 
    cuda_sync_and_check()

    // copy the 0th element out of the vector now that it contains the sum
    _cudaStatus = cudaMemcpy(out, _vec,sizeof(double), cudaMemcpyDeviceToHost);
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! area\n");
    throw;
    }


    return *out;

}

void DeviceMesh::cuda_sync_and_check(){
    _cudaStatus = cudaGetLastError();
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(_cudaStatus));
        throw;
    }
    // check that the kernal didn't throw an error
    _cudaStatus = cudaDeviceSynchronize();
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error %s after launching Kernel!\n", cudaGetErrorString(_cudaStatus));
        throw;
    }

}

void DeviceMesh::decend_gradient(Gradient *myGrad,double lambda){

    unsigned int numberOfBlocks = ceil(_numFacets / (float) _blockSize);

    // call vector add kerenal with force pointer and vertex pointer
    addWithMultKernel<<<numberOfBlocks,_blockSize>>>(_vert ,myGrad->get_force(),lambda)
    cuda_sync_and_check()

}

Gradient::Gradient(DeviceMesh *inMesh){
    _myMesh = inMesh;
    unsigned int numVert = _myMesh->get_numVert();
    unsigned int numFacet = _myMesh->get_numFacet();

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
        _cudaStatus = cudaMalloc((void**)&_gradVFacet, numVert * 3 * sizeof(double));
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }


        _cudaStatus = cudaMalloc((void**)&_force, numVert * 3 * sizeof(double));
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
};

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
    areaGradient<<<numberOfBlocks,_myMesh->get_blockSize()>>>(_gradAFacet,_myMesh->get_facets(),_myMesh->get_verts());
    cuda_sync_and_check();

    facet_to_vertex(_gradAFacet,_gradAVert);

}

void Gradient::calc_gradV(){
    // first calculate the gradient on the facets
    unsigned int numberOfBlocks = ceil(_myMesh->get_numFacets() / (float) _myMesh->get_blockSize());
    volumeGradient<<<numberOfBlocks,_myMesh->get_blockSize()>>>(_gradVFacet,_myMesh->get_facets(),_myMesh->get_verts());
    cuda_sync_and_check();

    facet_to_vertex(_gradVFacet,_gradVVert);

}

void Gradient::facet_to_vertex(_facetValue,_vertexValue){

    unsigned int numberOfBlocks = ceil(_myMesh->get_numVert() / (float) _myMesh->get_blockSize());
    facetToVertex<<<<<<numberOfBlocks,_myMesh->get_blockSize()>>>(_vertexValue,_facetValue,_myMesh->get_vertToFacet(), _myMesh->get_vertIndexStart)
    cuda_sync_and_check();

}
void Gradient::project_to_force(){

    unsigned int numberOfBlocks = ceil(_myMesh->get_numVert() / (float) _myMesh->get_blockSize());
    projectForce<<<numberOfBlocks,_myMesh->get_blockSize()>>>(_force,_gradAVert,_gradVVert);
    cuda_sync_and_check();

}

void Gradient::cuda_sync_and_check(){
    _cudaStatus = cudaGetLastError();
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(_cudaStatus));
        throw;
    }
    // check that the kernal didn't throw an error
    _cudaStatus = cudaDeviceSynchronize();
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error %s after launching Kernel!\n", cudaGetErrorString(_cudaStatus));
        throw;
    }

}