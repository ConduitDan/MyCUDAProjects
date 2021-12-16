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

Mesh::Mesh(unsigned int numVertIn,unsigned int numFacetsIn,double * vertIn,unsigned int* facetsIn){
    _numVert = numVertIn;
	_numFacets = numFacetsIn;
	_vert = vertIn;
	_facets = facetsIn;
}


Mesh::~Mesh(){

    if (_vert) delete[] _vert;
    if (_facets) delete[] _facets;
}

bool Mesh::operator ==(const Mesh& rhs){
    double tol = 1e-4;// This could be smaller
    
    if (_numVert!=rhs._numVert){
        return false;
    }
    
    if (_numFacets!=rhs._numFacets){
        return false;
    }
    
    
    for (int i = 0; i<_numVert*3; i++){
        if (abs((_vert[i]-rhs._vert[i])>tol)){
            return false;
        }
    }
    for (int i = 0; i<_numFacets*3; i++){
        if (_facets[i]!=rhs._facets[i]){
            return false;
        }
    }
    return true;

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
    _cudaStatus = cudaMalloc((void**)&_vertIndexStart, (_numVert+1) * sizeof(unsigned int));
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
    _cudaStatus = cudaMemcpy(_vertIndexStart, vertIndexStart, (_numVert+1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! vertices\n");
    }




    delete[] vertToFacet;
    delete[] vertIndexStart;
    delete[] vertCount;

}

DeviceMesh::~DeviceMesh(){
    if (_vert) {
	    cudaFree(_vert);
	    _vert = nullptr;
    }
    if (_facets){
	    cudaFree(_facets);
	    _facets = nullptr;
    }
    if (_vertToFacet){
	    cudaFree(_vertToFacet);
	    _vertToFacet = nullptr;
    }
    if (_vertIndexStart){
	    cudaFree(_vertIndexStart);
	    _vertIndexStart = nullptr;
    }
    if (_area){
	    cudaFree(_area);
	    _area = nullptr;
    }
    //if (_areaSum) cudaFree(_areaSum);
    if (_volume){
	    cudaFree(_volume);
	    _volume = nullptr;
    }
    //if (_volumeSum) cudaFree(_volumeSum);
}

Mesh DeviceMesh::copy_to_host(){
    //Mesh newMesh;
    double* newVert = new double[_numVert * 3]; // [x0; y0; z0; x1; y1;.... ]
    unsigned int* newFacets =  new unsigned int[_numFacets * 3];// [a0; b0; c0; a1;b1;c1;...]

    _cudaStatus = cudaMemcpy(newVert, _vert, _numVert * 3 *  sizeof(double), cudaMemcpyDeviceToHost);
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
    }

    _cudaStatus = cudaMemcpy(newFacets, _facets, _numFacets * 3 *  sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
    }
    return Mesh( _numVert, _numFacets,newVert,newFacets);



}

double DeviceMesh::area(){

    unsigned int numberOfBlocks = ceil(_numFacets / (float) _blockSize);
    areaKernel<<<numberOfBlocks, _blockSize>>> (_area, _vert, _facets, _numFacets);
    cuda_sync_and_check(_cudaStatus,"area");
    return sum_of_elements(_cudaStatus,_area,_numFacets,_bufferedSize,_blockSize);

}

double DeviceMesh::volume(){
    unsigned int numberOfBlocks = ceil(_numFacets / (float) _blockSize);

    volumeKernel<<<numberOfBlocks, _blockSize>>> (_volume, _vert, _facets, _numFacets);

    cuda_sync_and_check(_cudaStatus,"volume");
    return sum_of_elements(_cudaStatus,_volume,_numFacets,_bufferedSize,_blockSize);

}


double* DeviceMesh::check_area_on_facet(){
    unsigned int numberOfBlocks = ceil(_numFacets / (float) _blockSize);
    areaKernel<<<numberOfBlocks, _blockSize>>> (_area, _vert, _facets, _numFacets);
    cuda_sync_and_check(_cudaStatus,"area");

    double *areaPerFacet =  new double[_numFacets];

    _cudaStatus = cudaMemcpy(areaPerFacet, _area, _numFacets  *  sizeof(double), cudaMemcpyDeviceToHost);
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
    }
    return areaPerFacet;

}




void DeviceMesh::decend_gradient(Gradient *myGrad,double lambda){

    unsigned int numberOfBlocks = ceil(_numVert*3 / (float) _blockSize);

    // call vector add kerenal with force pointer and vertex pointer
    addWithMultKernel<<<numberOfBlocks,_blockSize>>>(_vert ,myGrad->get_force(),lambda,_numVert*3);
    cuda_sync_and_check(_cudaStatus,"add with scale");

}

