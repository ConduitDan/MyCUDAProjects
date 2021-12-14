//Tests.cpp
//here are some tests

#include "Mesh.hpp"
#include <iostream>
#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

//namespace bt = boost::unit_test;



//#####################
// mesh tests
//#####################
// constuctors
// read in from file

// given pointers

//#####################
// device mesh tests
//#####################

// constuct from mesh

// calc area

BOOST_AUTO_TEST_CASE(area){
    //area test
    Mesh square = Mesh("square.mesh");
    DeviceMesh dSquare = DeviceMesh(&square,128);
    double area = dSquare.area();
    std::cout<<"Testing area of a square\n";
    BOOST_CHECK (area == 1.0);
}
// calc volume
//


//#####################
// Kernal Tests
//#####################

// sum_of_elements
// dot product
// add with MultKernal
// area
// areaGrad
BOOST_AUTO_TEST_CASE(areaGrad){


    cudaError cudaStatus;

    // expected Vals

    double firstEle[14] = { -1, 1, -1, 1, -1, 1, -1, 1, 0, 0, 0, 0, 0, 0};
    double secondEle[14] = { -1, -1, 1, 1, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0};
    double thirdEle[14] = { -1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0};
    
    Mesh myMesh = Mesh("cube.mesh");
    DeviceMesh myDMesh = DeviceMesh(&myMesh,128);
    Gradient myGrad = Gradient(&myDMesh);
    myGrad.calc_force();

    double * gradA = new double[14*3];
    cudaStatus = cudaMemcpy(gradA, myGrad.get_gradA(), 14 * 3 *  sizeof(double), cudaMemcpyDeviceToHost);
    BOOST_REQUIRE (cudaStatus == cudaSuccess);

    double tol = 1e-6;
    for (int i = 0; i<14; i++){
        BOOST_CHECK(abs(firstEle[i]-gradA[i*3])<tol);
        BOOST_CHECK(abs(secondEle[i]-gradA[i*3+1])<tol);
        BOOST_CHECK(abs(thirdEle[i]-gradA[i*3+2])<tol);

    }
}

// volume
// volumeGrad

BOOST_AUTO_TEST_CASE(volumeGrad){

    cudaError cudaStatus;

    // expected Vals
    double firstEle[14] = {-0.166667, 0.166667, -0.166667, 0.166667, -0.166667, 0.166667, -0.166667, 0.166667, 0, 0, 0, 0, -0.333333, 0.333333};
    double secondEle[14] = { -0.166667, -0.166667, 0.166667, 0.166667, -0.166667, -0.166667, 0.166667, 0.166667, 0, 0, -0.333333, 0.333333, 0, 0 };
    double thirdEle[14] = { -0.166667, -0.166667, -0.166667, -0.166667, 0.166667, 0.166667, 0.166667, 0.166667, -0.333333, 0.333333, 0, 0, 0, 0 };
    
    Mesh myMesh = Mesh("cube.mesh");
    DeviceMesh myDMesh = DeviceMesh(&myMesh,128);
    Gradient myGrad = Gradient(&myDMesh);
    myGrad.calc_force();

    double * gradV = new double[14*3];
    cudaStatus = cudaMemcpy(gradV, myGrad.get_gradV(), 14 * 3 *  sizeof(double), cudaMemcpyDeviceToHost);
    BOOST_REQUIRE (cudaStatus == cudaSuccess);

    double tol = 1e-4;
    for (int i = 0; i<14; i++){
        BOOST_CHECK(abs(firstEle[i]-gradV[i*3])<tol);
        BOOST_CHECK(abs(secondEle[i]-gradV[i*3+1])<tol);
        BOOST_CHECK(abs(thirdEle[i]-gradV[i*3+2])<tol);

    }

    delete[] gradV;
}





// face to Vertex
BOOST_AUTO_TEST_CASE(face_to_vertex){
    //set up some verteices
    // x--x--x
    //  \/ \/
    //  x --x
    cudaError cudaStatus;


    unsigned int numVert = 5;
    unsigned int numFace = 3;
    unsigned int blockSize = 128;

    double *vert = new double[15] {0.5,0,0,
                        1.5,0,0,
                        0,1,0,
                        1,1,0,
                        2,1,0};

    // set up some faces
    unsigned int *faces = new unsigned int[9] {0,2,3,
                             0,1,3,
                             1,3,4};
    // set up some values on the faces
    //each face gets a 3 vectors
    
    // create a mesh from it and then put it on the GPU
    Mesh myMesh = Mesh(numVert,numFace,vert,faces);
    DeviceMesh myDMesh = DeviceMesh(&myMesh,blockSize);
    // check that the maps are create correctly

    unsigned int *vertToFacet = new unsigned int[numFace * 3] ;
    cudaStatus = cudaMemcpy(vertToFacet, myDMesh.get_vertToFacet(), numFace * 3 *  sizeof(unsigned int), cudaMemcpyDeviceToHost);
    BOOST_REQUIRE (cudaStatus == cudaSuccess);

    unsigned int *vertToFacetStart = new unsigned int[(numVert + 1)];
    cudaStatus = cudaMemcpy(vertToFacetStart, myDMesh.get_vertIndexStart(), (numVert + 1) *  sizeof(unsigned int), cudaMemcpyDeviceToHost);
    BOOST_REQUIRE (cudaStatus == cudaSuccess);

    myDMesh.get_vertIndexStart();

    unsigned int index;
    for (int i = 0; i< numVert; i++){
        for (int j = vertToFacetStart[i]; j< vertToFacetStart[i+1];j++){
            index = vertToFacet[j];
            BOOST_CHECK(faces[index]==i);
        }
    }


    // create the expected answer
    double *expectedVertVals = new double[5*3];
    for (int i = 0; i< numVert*3; i++){
        expectedVertVals[i] = 0;
    }

    double *faceValues = new double[27];
    for (int i = 0; i<9; i++){
        faceValues[i*3]=(double)i/10.0;
        faceValues[i*3+1]=(double)i/10.0;
        faceValues[i*3+2]=(double)i/10.0;

        // calculate the expected answer
        expectedVertVals[faces[i]*3] += (double)i/10.0;
        expectedVertVals[faces[i]*3+1] += (double)i/10.0;
        expectedVertVals[faces[i]*3+2] += (double)i/10.0;
    }
    double *dfaceValues = nullptr;
    cudaStatus = cudaMalloc((void**)&dfaceValues, numFace * 9 * sizeof(double));
    BOOST_REQUIRE (cudaStatus == cudaSuccess);
    cudaStatus = cudaMemcpy(dfaceValues, faceValues, numFace * 9 * sizeof(double), cudaMemcpyHostToDevice);
    BOOST_REQUIRE (cudaStatus == cudaSuccess);
    


    double *vertValues = nullptr;
    // allocate space for the values on the vertex
    cudaStatus = cudaMalloc((void**)&vertValues, numVert * 3 * sizeof(double));
    BOOST_REQUIRE (cudaStatus == cudaSuccess);
    unsigned int numberOfBlocks = ceil(numVert/(float)blockSize)*blockSize;
    // calculate face to vertex
    facetToVertex<<<numberOfBlocks, blockSize>>>(vertValues,dfaceValues,myDMesh.get_vertToFacet(), myDMesh.get_vertIndexStart(),numVert);

    // copy back
    double *actualVertVals = new double[5*3];
    cudaStatus = cudaMemcpy(actualVertVals, vertValues, numVert * 3 *  sizeof(double), cudaMemcpyDeviceToHost);
    BOOST_REQUIRE (cudaStatus == cudaSuccess);


    // does it match expectation?
    for (int i = 0; i<numVert*3; i++){
        BOOST_CHECK(actualVertVals[i]==expectedVertVals[i]);
        if (actualVertVals[i]!=expectedVertVals[i]){
            printf("Element %d Failed, Expected %f\t found %f\n",i,expectedVertVals[i],actualVertVals[i]);
        }
    }


    cudaFree(vertValues);
    cudaFree(dfaceValues);
    delete[] actualVertVals;
    delete[] faceValues;



}
// project force
// element multiply







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
