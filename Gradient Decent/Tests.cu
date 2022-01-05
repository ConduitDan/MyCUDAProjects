//Tests.cpp
//here are some tests

#include "Mesh.hpp"
#include "ShapeOptimizer.hpp"
#include <iostream>
#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

//namespace bt = boost::unit_test;
//#####################
// Cube-> sphere test
//#####################
BOOST_AUTO_TEST_CASE(cube_to_sphere){
    ShapeOptimizer myOptimizer("testCube.mesh");
    myOptimizer.gradientDesent(12);

    Mesh acutal = myOptimizer.get_optimized_mesh();
    Mesh expected = Mesh("testCubeRelaxed.mesh");

    BOOST_CHECK(acutal==expected);

}

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
    DeviceMesh dSquare = DeviceMesh(&square,new CUDA(128));
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
    
	DeviceAPI* GPU_API = new CUDA(128);
    Mesh myMesh = Mesh("cube.mesh");
    DeviceMesh myDMesh = DeviceMesh(&myMesh,GPU_API);
    Gradient myGrad = Gradient(&myDMesh,GPU_API);
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
    
	DeviceAPI* GPU_API = new CUDA(128);
    Mesh myMesh = Mesh("cube.mesh");
    DeviceMesh myDMesh = DeviceMesh(&myMesh,GPU_API);
    Gradient myGrad = Gradient(&myDMesh,GPU_API);
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
    DeviceMesh myDMesh = DeviceMesh(&myMesh,new CUDA(blockSize));
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
BOOST_AUTO_TEST_CASE(project_force){


    // expected Vals
   // double firstEle[14] = { 0.42265, -0.42265, 0.42265, -0.42265, 0.42265, -0.42265, 0.42265, -0.42265, 0, 0, 0, 0, -1.1547, 1.1547 };
   // double secondEle[14] = { 0.42265, 0.42265, -0.42265, -0.42265, 0.42265, 0.42265, -0.42265, -0.42265, 0, 0, -1.1547, 1.1547, 0, 0 };
   // double thirdEle[14] = { 0.42265, 0.42265, 0.42265, 0.42265, -0.42265, -0.42265, -0.42265, -0.42265, -1.1547, 1.1547, 0, 0, 0, 0 };
    double firstEle[14] =  { 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0, 0, 0, 0, -1, 1};
    double secondEle[14] = { 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0, 0, -1, 1, 0, 0};
    double thirdEle[14] = { 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -1, 1, 0, 0, 0, 0};
    

    double tol = 1e-4;

	DeviceAPI* GPU_API = new CUDA(128);
    Mesh myMesh = Mesh("cube.mesh");
    DeviceMesh myDMesh = DeviceMesh(&myMesh,GPU_API);
    Gradient myGrad = Gradient(&myDMesh,GPU_API);

    myGrad.calc_force();

    double * force = new double[14*3];
    cudaError cudaStatus = cudaMemcpy(force, myGrad.get_force(), 14 * 3 *  sizeof(double), cudaMemcpyDeviceToHost);
    BOOST_REQUIRE (cudaStatus == cudaSuccess);

    double * gradV = new double[14*3];
    cudaStatus = cudaMemcpy(gradV, myGrad.get_gradV(), 14 * 3 *  sizeof(double), cudaMemcpyDeviceToHost);
    BOOST_REQUIRE (cudaStatus == cudaSuccess);

    double * gradA = new double[14*3];
    cudaStatus = cudaMemcpy(gradA, myGrad.get_gradA(), 14 * 3 *  sizeof(double), cudaMemcpyDeviceToHost);
    BOOST_REQUIRE (cudaStatus == cudaSuccess);


    double GAGV = 0;
    double GVGV = 0;
    for (int i = 0; i<14*3; i++){
        GAGV += gradA[i]*gradV[i];
        GVGV += gradV[i]*gradV[i];
    }

    BOOST_CHECK(abs(GAGV-4)<tol);
    BOOST_CHECK(abs(GVGV-1.33333)<tol);

    for (int i = 0; i<14; i++){
        BOOST_CHECK(abs(firstEle[i]-force[i*3])<tol);
        BOOST_CHECK(abs(secondEle[i]-force[i*3+1])<tol);
        BOOST_CHECK(abs(thirdEle[i]-force[i*3+2])<tol);
    }

    delete[] force;
    delete[] gradA;
    delete[] gradV;


}
// element multiply


