//Tests.cpp
//here are some tests without a harness


#include "Mesh.hpp"
#include <iostream>
#define BOOST_TEST_MAIN
#include "boost/test/unit_test.hpp"
namespace bt = boost::unit_test;



    //#####################
    // mesh tests
    //#####################

BOOST_AUTO_TEST_CASE(area){
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
}
/*
    //###########################
    // kernal tests
    //###########################

    //add tree
    unsigned int blockSize = 4;
    unsigned int size = 8;
    double* array = new double[size];
    for (int i = 0; i <size; i++){
        array[i] = i;
    }



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
    
    double out = sum_of_elements(_cudaStatus, dArray, size, _bufferedSize, blockSize);
    // copy the 0th element out of the vector now that it contains the sum
    cudaFree(dArray);


    int expected = size*(size-1)/2;
    std::cout<<"Testing addtree \n";
    std::cout<<"Expected value: "<<expected<<"\t Acutal value: "<<out<<std::endl;
    if (out == expected) std::cout<<"Passed\n";
    else std::cout<<"Failed\n";


    size = 10;
    _bufferedSize = ceil(size / (float)( blockSize * 2)) * 2 * blockSize; // for 

    // Dot product
    double* array1 = new double[size];
    double* array2 = new double[size];
    for (int i = 0; i <size; i++){
        array1[i] = i;
        array2[i] = 2;
    }
    double *dArray1 = nullptr;
    double *dArray2 = nullptr;
    double *dscratch = nullptr;

    _cudaStatus = cudaMalloc((void**)&dArray1, size * sizeof(double));
    _cudaStatus = cudaMalloc((void**)&dArray2, size * sizeof(double));
    _cudaStatus = cudaMalloc((void**)&dscratch, _bufferedSize * sizeof(double));
    
    _cudaStatus = cudaMemcpy(dArray1,array1, size * sizeof(double), cudaMemcpyHostToDevice);
    _cudaStatus = cudaMemcpy(dArray2,array2, size * sizeof(double), cudaMemcpyHostToDevice);

    double ans = dotProduct(_cudaStatus,dArray1,dArray2,dscratch,size,blockSize);
    expected = size*(size-1);
    std::cout<<"Testing dot product \n";
    std::cout<<"Expected value: "<<expected<<"\t Acutal value: "<<ans<<std::endl;
    if (expected == ans) std::cout<<"Passed\n";
    else std::cout<<"Failed\n";

    cudaFree(dArray1);
    cudaFree(dArray2);
    cudaFree(dscratch);




}*/