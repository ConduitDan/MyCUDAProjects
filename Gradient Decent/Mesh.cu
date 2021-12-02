#include "Mesh.hpp"

Mesh::Mesh(){
    _numVert = 0;
	_numFacets = 0;
	_vert = nullptr;
	_facets = nullptr;

}
Mesh::Mesh(const char* fileName){
    if(load_mesh_from_file(fileName)){
        fprintf(stdout,"Successfully read %s \n",fileName);
        fprintf(stdout,"Found %d vertices and %d faces\n",_numVert,_numFacets);
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
        numAssigned = fscanf(fp, "%*d %d %d %d\n", (_facets + i * 3), (_facets + i * 3 + 1), (_facets + i * 3 + 2));
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
bool Mesh::print(const char* fileName){
    // open file
    FILE* fp;
    fp = fopen(fileName, "w");
    if (fp == NULL) {
        fprintf(stderr, "Could not open file\n");
        return false;
    }

    // print vertices
    fprintf(fp,"vertices\n\n");
    for (int i = 0; i<_numVert; i++){
        fprintf(fp,"%d %f %f %f\n",i+1,_vert[i*3],_vert[i*3+1],_vert[i*3+2]);
    }

    // print facets
    fprintf(fp,"\nfaces\n\n");
    for (int i = 0; i<_numFacets; i++){
        fprintf(fp,"%d %d %d %d\n",i+1,_facets[i*3]+1,_facets[i*3+1]+1,_facets[i*3+2]+1);
    }
    return 1;
}



DeviceMesh::DeviceMesh(Mesh* hostMesh, unsigned int blockSize){
    // set up the GPU 
    _cudaStatus = cudaSetDevice(0);
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    
    
    // copy over the number of elements
    _numFacets = hostMesh->get_numFacets();
    _numVert = hostMesh->get_numVert();

    // create the vertex-><facets, # in facet> map
    // there should be 3*_numfacets values in the map.
    // we should store a compaion array with the starting index for vertex i in the map

    unsigned int *vertToFacet  = new unsigned int [_numFacets * 3] {0};// (store i*3 + j);
    //unsigned int *vertToFacetIndex = new unsigned int [_numFacets * 3] {0};
    unsigned int *vertIndexStart = new unsigned int [_numVert+1] {0}; // + one more here so we have[0...numVert] to make logic simpler for first or last ele
    unsigned int *vertCount = new unsigned int [_numVert] {0};

    // TODO: memset here
    
    unsigned int* hostFacets = hostMesh->get_facets();

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


    _bufferedSize = ceil(_numFacets / (float)( blockSize * 2)) * 2 * blockSize; // for 

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
    _cudaStatus = cudaMalloc((void**)&_area, _bufferedSize * sizeof(double));
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    _cudaStatus = cudaMalloc((void**)&_volume, _bufferedSize * sizeof(double));
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    


    // copy over the vertices and facets
    _cudaStatus = cudaMemcpy(_vert, hostMesh->get_vert(), _numVert * 3 * sizeof(double), cudaMemcpyHostToDevice);
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! vertices\n");
    }
    _cudaStatus = cudaMemcpy(_facets, hostMesh->get_facets(), _numFacets * 3 * sizeof(unsigned int), cudaMemcpyHostToDevice);
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
    newMesh._facets =  new unsigned int[_numFacets * 3];// [a0; b0; c0; a1;b1;c1;...]

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
    cuda_sync_and_check("area");
    return sum_of_elements(_area,_numFacets);

}

double DeviceMesh::volume(){
    unsigned int numberOfBlocks = ceil(_numFacets / (float) _blockSize);

    volumeKernel<<<numberOfBlocks, _blockSize>>> (_volume, _vert, _facets, _numFacets);

    cuda_sync_and_check("volume");
    return sum_of_elements(_volume,_numFacets);

}

double DeviceMesh::sum_of_elements(double* vec,unsigned int size){

    double out;



    // do the reduction each step sums _blockSize*2 number of elements
    unsigned int numberOfBlocks = ceil(size / (float) _blockSize / 2.0);
    // printf("AddTree with %d blocks,  of blocks size %d, for %d total elements\n",numberOfBlocks,_blockSize,_bufferedSize);
    
    addTree<<<numberOfBlocks, _blockSize, _bufferedSize / 2 * sizeof(double) >>> (vec, vec);

    
    // reduce6<128><<< numberOfBlocks, _blockSize, _bufferedSize / 2 * sizeof(double)>>>(vec, vec,_bufferedSize);

        
    // double *check = new double[_bufferedSize];
    // _cudaStatus = cudaMemcpy(check, vec,sizeof(double)*_bufferedSize, cudaMemcpyDeviceToHost);
    // if (_cudaStatus != cudaSuccess) {
    //     fprintf(stderr, "cudaMemcpy failed! area\n");
    // throw;
    // }
    // for (int i = 0; i<_bufferedSize; i++){
    //     printf("i = %d \t val = %f\n",i,check[i]);
    // }
    if (numberOfBlocks>1){
        for (int i = numberOfBlocks; i > 1; i /= (_blockSize * 2)) {
            addTree<<<ceil((float)numberOfBlocks/ (_blockSize * 2)), _blockSize, ceil((float)size / 2)* sizeof(double) >>> (vec, vec);
        } 
    }
    cuda_sync_and_check("sum of elements");

    // copy the 0th element out of the vector now that it contains the sum
    _cudaStatus = cudaMemcpy(&out, vec,sizeof(double), cudaMemcpyDeviceToHost);
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! area\n");
    throw;
    }




    return out;

}
double* DeviceMesh::check_area_on_facet(){
    unsigned int numberOfBlocks = ceil(_numFacets / (float) _blockSize);
    areaKernel<<<numberOfBlocks, _blockSize>>> (_area, _vert, _facets, _numFacets);
    cuda_sync_and_check("area");

    double *areaPerFacet =  new double[_numFacets];

    _cudaStatus = cudaMemcpy(areaPerFacet, _area, _numFacets  *  sizeof(double), cudaMemcpyDeviceToHost);
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
    }
    return areaPerFacet;

}


void DeviceMesh::cuda_sync_and_check(const char * caller){
    _cudaStatus = cudaGetLastError();
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(_cudaStatus));
        throw;
    }
    // check that the kernal didn't throw an error
    _cudaStatus = cudaDeviceSynchronize();
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error %s after launching Kernel %s!\n", cudaGetErrorString(_cudaStatus),caller);
        throw;
    }

}

void DeviceMesh::decend_gradient(Gradient *myGrad,double lambda){

    unsigned int numberOfBlocks = ceil(_numFacets / (float) _blockSize);

    // call vector add kerenal with force pointer and vertex pointer
    addWithMultKernel<<<numberOfBlocks,_blockSize>>>(_vert ,myGrad->get_force(),lambda,_numVert);
    cuda_sync_and_check("add with scale");

}

