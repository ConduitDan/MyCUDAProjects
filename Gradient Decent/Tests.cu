//Tests.cpp
//here are some tests without a harness


#include "Mesh.hpp"
#include <iostream>

int main(){

    //#####################
    // mesh tests
    //#####################

    //area test
    
    Mesh square = Mesh("square.mesh");
    square.print("StillASquare.mesh");
    DeviceMesh dSquare = DeviceMesh(&square,128);
    double area = dSquare.area();
    std::cout<<"Testing area of a square\n";
    std::cout<<"Expected value: 1\t Acutal value: "<<area<<std::endl;
    if (area == 1.0) std::cout<<"Passed\n";
    else std::cout<<"Failed\n";


    double * areaPerFacet = dSquare.check_area_on_facet();
    for (int i = 0; i< dSquare.get_numFacets(); i++){
       std::cout<<"Facet "<<i+1<<" has area: "<<areaPerFacet[i]<<std::endl;
    }

    //###########################
    // kernal tests
    //###########################



    //add tree
    unsigned int blockSize = 4;
    unsigned int size = 8;
    double array[8] = {1,2,3,4,5,6,7,8};

    double *dArray = nullptr;
    
    unsigned int _bufferedSize = ceil(size / (float)( blockSize * 2)) * 2 * blockSize; // for 
    cudaError_t _cudaStatus;

    //allocate
    _cudaStatus = cudaMalloc((void**)&dArray, _bufferedSize * sizeof(double));
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    // copy over
    _cudaStatus = cudaMemcpy(dArray,array, _bufferedSize * sizeof(double), cudaMemcpyHostToDevice);
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! vertices\n");
    }

    //add
    unsigned int numberOfBlocks = ceil(size / (float) blockSize / 2.0);
    addTree<<<numberOfBlocks,blockSize, _bufferedSize / 2 * sizeof(double) >>> (dArray, dArray,_bufferedSize);
    if (numberOfBlocks>1){
        for (int i = numberOfBlocks; i > 1; i /= (blockSize * 2)) {
        addTree<<<ceil((float)numberOfBlocks/ (blockSize * 2)), blockSize, ceil((float)size / 2)* sizeof(double) >>> (dArray, dArray,_bufferedSize);
        } 
    }
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
    double out = 0;
    // copy the 0th element out of the vector now that it contains the sum
    _cudaStatus = cudaMemcpy(&out, dArray,sizeof(double), cudaMemcpyDeviceToHost);
    if (_cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! area\n");
    throw;
    }

    cudaFree(dArray);


    int expected = size*(size+1)/2;;
    std::cout<<"Testing addtree \n";
    std::cout<<"Expected value: "<<expected<<"\t Acutal value: "<<out<<std::endl;
    if (out == expected) std::cout<<"Passed\n";
    else std::cout<<"Failed\n";




}