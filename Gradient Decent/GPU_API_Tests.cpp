//CUDATests

//#include "Mesh.hpp"
#include "ShapeOptimizer.hpp"
#include "Mesh.hpp"

#include <iostream>
#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>
#include "APIFactory.hpp"

#define BLOCKSIZE 64

// THings aren't working because the unique device pointers are not being passed by reference





BOOST_AUTO_TEST_CASE(Device_Pointers){
	//copy an array the the device and copy it back 
	DeviceAPI* GPU = APIFactory::get_API(BLOCKSIZE);
	printf("made GPU API\n");
	double myArray[5] = {1,2,3,4,5};

	UniqueDevicePtr<double>devArray(GPU);
	printf("initalized DevicePointer\n");
	devArray.allocate(5);
	printf("allocated Device Pointer\n");
	GPU->copy_to_device(devArray.get(),myArray,5*sizeof(double));
	printf("Copied data to device\n");
	double myArrayCopiedBack[5];

	GPU->copy_to_host(myArrayCopiedBack,devArray.get(),5*sizeof(double));
	printf("Copied data from device\n");
	for (int i = 0; i<5; i++){
		BOOST_CHECK(myArrayCopiedBack[i]==myArray[i]);
		if (myArrayCopiedBack[i]!=myArray[i]){
			printf("%f (actual) != %f (expected)\n",myArrayCopiedBack[i],myArray[i]);
		}
	}
}



BOOST_AUTO_TEST_CASE(add_with_mult){
	
	DeviceAPI* GPU = APIFactory::get_API(BLOCKSIZE);

	double myArray[5] = {1,2,3,4,5};
	double myArrayCopiedBack[5] = {0,0,0,0,0};

	UniqueDevicePtr<double>a(GPU);
	UniqueDevicePtr<double>b(GPU);


	a.allocate(5);
	b.allocate(5);

	GPU->copy_to_device(a.get(),myArray,5*sizeof(double));
	GPU->copy_to_device(b.get(),myArray,5*sizeof(double));

	GPU->add_with_mult(&a,&b, 2.0, 5); 
	
	GPU->copy_to_host(myArrayCopiedBack,a.get(),5*sizeof(double));


	for (int i = 0; i<5; i++){
		BOOST_CHECK(myArrayCopiedBack[i]==(i+1)*3);
		if (myArrayCopiedBack[i]!=(i+1)*3){
			printf("%f (actual) != %f (expected)\n",myArrayCopiedBack[i],(double)(i+1)*3);
		}
	}

}


BOOST_AUTO_TEST_CASE(sum_of_elements){
	DeviceAPI* GPU = APIFactory::get_API(BLOCKSIZE);
	double myArray[5] = {1,2,3,4,5};

	UniqueDevicePtr<double>* vec = new UniqueDevicePtr<double>(GPU);
	vec->allocate(128*2);
	
	GPU->copy_to_device(vec->get(),myArray,5*sizeof(double));
	int size = 5;
	int bufferedSize = 128*2;

	double ans = GPU->sum_of_elements(vec, size,bufferedSize);
	delete vec;
	BOOST_CHECK(ans==1+2+3+4+5);
	if (ans !=1+2+3+4+5){
		printf("expected %d, got %f\n",1+2+3+4+5,ans);
	}



}






BOOST_AUTO_TEST_CASE(dotProduct){
	DeviceAPI* GPU = APIFactory::get_API(BLOCKSIZE);
	double myArray[5] = {1,2,3,4,5};

	UniqueDevicePtr<double> v1(GPU);
	UniqueDevicePtr<double> v2(GPU);
	UniqueDevicePtr<double> scratch(GPU);

	v1.allocate(5);
	v2.allocate(5);
	scratch.allocate(128*2);

	GPU->copy_to_device(v1.get(),myArray,sizeof(double)*5);
	GPU->copy_to_device(v2.get(),myArray,sizeof(double)*5);
	int size = 5;

	double out = GPU->dotProduct(&v1,&v2, &scratch, size);
	BOOST_CHECK(out == 1+4+9+16+25);


}


BOOST_AUTO_TEST_CASE(project_force){
	// calculates c = -(a - scale * b)

	DeviceAPI* GPU = APIFactory::get_API(BLOCKSIZE);


	UniqueDevicePtr<double> a(GPU);
	UniqueDevicePtr<double> b(GPU);
	UniqueDevicePtr<double> c(GPU);

	a.allocate(5);
	b.allocate(5);
	c.allocate(5);

	double myA[5] = {1,2,3,4,5};
	double myB[5] = {1,1,1,1,1};
	double scale = 3;
	GPU->copy_to_device(a.get(),myA,5*sizeof(double));
	GPU->copy_to_device(b.get(),myB,5*sizeof(double));

	
	//({1,2,3,4,5}- (1+2+3+4+5)/5 {1,1,1,1,1})
	// ({1,2,3,4,5}-{3,3,3,3,3})
	// ({-2,-1,0,1,2})
	double expected[5] = {2,1,0,-1,-2};
	double actual[5];

	GPU->project_force(&c, &a, &b, scale,5);
	GPU->copy_to_host(actual,c.get(),5*sizeof(double));

	for (int i = 0; i<5; i++){

		//printf("expected %f, got %f .... delta = %f\n",expected[i],actual[i],expected[i]-actual[i]);
		BOOST_CHECK(expected[i]==actual[i]);
	}

}


BOOST_AUTO_TEST_CASE(facet_to_vertex){
	// lets do this with 3 facets
	//		3-----4  
	//     / \  /  \
	//    /   \/    \
	//   0 --- 1 --- 2
	// facetList = { 0, 3, 1,
	//				 1, 4, 2,
	//				 3, 4, 1}
	//}

	DeviceAPI* GPU = APIFactory::get_API(BLOCKSIZE);

	int numFacet = 3;
	int numVert = 5;

	unsigned int vertToFacetHost[9] = {0,\
									   2,3,8,\
									   5,\
									   1,6,\
									   4,7};
	unsigned int vertIndexStartHost[6] = {0,1,4,5,7,9};

	double facetValueHost[27] = {1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9};
	double vertexValuesExpected[15] = {1,1,1,\
									   3+4+9,3+4+9,3+4+9,\
									   6,6,6,\
									   2+7,2+7,2+7,\
									   8+5,8+5,8+5 };

	UniqueDevicePtr<double> vertexValue(GPU);
	UniqueDevicePtr<double> facetValue(GPU);
	UniqueDevicePtr<unsigned int> vertToFacet(GPU);
	UniqueDevicePtr<unsigned int> vertIndexStart(GPU);

	vertexValue.allocate(15);

	facetValue.allocate(27);
	GPU->copy_to_device(facetValue.get(),facetValueHost,sizeof(double)*27);

	vertToFacet.allocate(9);
	GPU->copy_to_device(vertToFacet.get(),vertToFacetHost,sizeof(unsigned int)*9);

	vertIndexStart.allocate(9);
	GPU->copy_to_device(vertIndexStart.get(),vertIndexStartHost,sizeof(unsigned int)*6);


	GPU->facet_to_vertex( &vertexValue, &facetValue, &vertToFacet,&vertIndexStart,numVert);

	double ans[15];
	GPU->copy_to_host(ans,vertexValue.get(),sizeof(double)*15);

	for (int i=0; i<15; i++){
		BOOST_CHECK(ans[i]==vertexValuesExpected[i]);
		if (ans[i] !=vertexValuesExpected[i]){
		printf("expected %f, got %f\n",vertexValuesExpected[i],ans[i]);
	}

	}

}


BOOST_AUTO_TEST_CASE(facet_to_vertex_simple){
	// lets do this with 3 facets
	//		1
	//     / \ 
	//    /   \
	//   0 --- 2
	// facetList = { 0, 1, 2}
	//

	DeviceAPI* GPU = APIFactory::get_API(BLOCKSIZE);

	int numFacet = 1;
	int numVert = 3;

	unsigned int vertToFacetHost[3] = {0, 1, 2};
	unsigned int vertIndexStartHost[4] = {0,1,2,3};

	double facetValueHost[9] = {1,1,1,2,2,2,3,3,3};
	double vertexValuesExpected[9] = {1,1,1,\
									   2,2,2,\
									   3,3,3};

	UniqueDevicePtr<double> vertexValue(GPU);
	UniqueDevicePtr<double> facetValue(GPU);
	UniqueDevicePtr<unsigned int> vertToFacet(GPU);
	UniqueDevicePtr<unsigned int> vertIndexStart(GPU);

	vertexValue.allocate(9);

	facetValue.allocate(9);
	GPU->copy_to_device(facetValue.get(),facetValueHost,sizeof(double)*9);

	vertToFacet.allocate(3);
	GPU->copy_to_device(vertToFacet.get(),vertToFacetHost,sizeof(unsigned int)*3);

	vertIndexStart.allocate(4);
	GPU->copy_to_device(vertIndexStart.get(),vertIndexStartHost,sizeof(unsigned int)*4);


	GPU->facet_to_vertex( &vertexValue, &facetValue, &vertToFacet,&vertIndexStart,numVert);

	double ans[9];
	GPU->copy_to_host(ans,vertexValue.get(),sizeof(double)*9);

	for (int i=0; i<9; i++){
		BOOST_CHECK(ans[i]==vertexValuesExpected[i]);
		if (ans[i] !=vertexValuesExpected[i]){
		printf("expected %f, got %f\n",vertexValuesExpected[i],ans[i]);
	}

	}

}



/*
BOOST_AUTO_TEST_CASE(cube_to_sphere){
    ShapeOptimizer myOptimizer("Meshs/testCube.mesh");
    myOptimizer.gradientDesent(12);

    Mesh acutal = myOptimizer.get_optimized_mesh();
    Mesh expected = Mesh("Meshs/testCubeRelaxed.mesh");

    BOOST_CHECK(acutal==expected);

}*/
BOOST_AUTO_TEST_CASE(area){
    //area test
    Mesh square = Mesh("Meshs/square.mesh");
    DeviceMesh dSquare = DeviceMesh(&square,new CUDA(BLOCKSIZE));
    double area = dSquare.area();
    std::cout<<"Testing area of a square\n";
    BOOST_CHECK (area == 1.0);
	area = dSquare.area();
	BOOST_CHECK (area == 1.0);
	// make sure repeat calcuations are fine


}

BOOST_AUTO_TEST_CASE(areaGrad){

	DeviceAPI* GPU = APIFactory::get_API(BLOCKSIZE);


    // expected Vals

    double firstEle[14] = { -1, 1, -1, 1, -1, 1, -1, 1, 0, 0, 0, 0, 0, 0};
    double secondEle[14] = { -1, -1, 1, 1, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0};
    double thirdEle[14] = { -1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0};

    Mesh myMesh = Mesh("Meshs/cube.mesh");
    DeviceMesh myDMesh = DeviceMesh(&myMesh,GPU);
    Gradient myGrad = Gradient(&myDMesh,GPU);


    double * gradA = new double[14*3];
	for (int j = 0; j <3; j++){// do this 3x to make sure repeat calculations are fine
		myGrad.calc_gradA();
		GPU->copy_to_host(gradA, myGrad.get_gradA()->get(), 14 * 3 *  sizeof(double));
		
		double tol = 1e-6;
		for (int i = 0; i<14; i++){
			BOOST_CHECK(abs(firstEle[i]-gradA[i*3])<tol);
			if (abs(firstEle[i]-gradA[i*3])>tol){printf("vertex %d has bad x comp, expected %f, got %f\n",i,firstEle[i],gradA[i*3]);}
			BOOST_CHECK(abs(secondEle[i]-gradA[i*3+1])<tol);
			if (abs(secondEle[i]-gradA[i*3+1])>tol){printf("vertex %d has bad y comp, expected %f, got %f\n",i,secondEle[i],gradA[i*3+1]);}
			BOOST_CHECK(abs(thirdEle[i]-gradA[i*3+2])<tol);
			if (abs(thirdEle[i]-gradA[i*3+2])>tol){printf("vertex %d has bad z comp, expected %f, got %f\n",i,thirdEle[i],gradA[i*3+2]);}

		}
	}
}

BOOST_AUTO_TEST_CASE(volumeGrad){

	DeviceAPI* GPU = APIFactory::get_API(BLOCKSIZE);


    // expected Vals
    double firstEle[14] = {-0.166667, 0.166667, -0.166667, 0.166667, -0.166667, 0.166667, -0.166667, 0.166667, 0, 0, 0, 0, -0.333333, 0.333333};
    double secondEle[14] = { -0.166667, -0.166667, 0.166667, 0.166667, -0.166667, -0.166667, 0.166667, 0.166667, 0, 0, -0.333333, 0.333333, 0, 0 };
    double thirdEle[14] = { -0.166667, -0.166667, -0.166667, -0.166667, 0.166667, 0.166667, 0.166667, 0.166667, -0.333333, 0.333333, 0, 0, 0, 0 };
    
    Mesh myMesh = Mesh("Meshs/cube.mesh");
    DeviceMesh myDMesh = DeviceMesh(&myMesh,GPU);
    Gradient myGrad = Gradient(&myDMesh,GPU);
	double tol = 1e-6;

    double * gradV = new double[14*3];
	// for (int j = 0; j <3; j++){// do this 3x to make sure repeat calculations are fine

		myGrad.calc_gradV();
		GPU->copy_to_host(gradV, myGrad.get_gradV()->get(), 14 * 3 *  sizeof(double));
		for (int i = 0; i<14; i++){

			BOOST_CHECK(abs(firstEle[i]-gradV[i*3])<tol);
			if (abs(firstEle[i]-gradV[i*3])>tol){printf("vertex %d has bad x comp, expected %f, got %f\n",i,firstEle[i],gradV[i*3]);}
			BOOST_CHECK(abs(secondEle[i]-gradV[i*3+1])<tol);
			if (abs(secondEle[i]-gradV[i*3+1])>tol){printf("vertex %d has bad y comp, expected %f, got %f\n",i,secondEle[i],gradV[i*3+1]);}
			BOOST_CHECK(abs(thirdEle[i]-gradV[i*3+2])<tol);
			if (abs(thirdEle[i]-gradV[i*3+2])>tol){printf("vertex %d has bad z comp, expected %f, got %f\n",i,thirdEle[i],gradV[i*3+2]);}


		}
	// }

    delete[] gradV;
}








/*
BOOST_AUTO_TEST_CASE(area_gradient){
	area_gradient(UniqueDevicePtr<double> gradAFacet,UniqueDevicePtr<unsigned int> facets,UniqueDevicePtr<double> vert,unsigned int numFacets)
}
BOOST_AUTO_TEST_CASE(volume_gradient){
	volume_gradient(UniqueDevicePtr<double> gradVFacet,UniqueDevicePtr<unsigned int> facets,UniqueDevicePtr<double> vert,unsigned int numFacets)

}
BOOST_AUTO_TEST_CASE(area){
	area(UniqueDevicePtr<double> area, UniqueDevicePtr<double> vert, UniqueDevicePtr<unsigned int> facets, unsigned int numFacets)
}
BOOST_AUTO_TEST_CASE(volume){
	volume(UniqueDevicePtr<double> volume, UniqueDevicePtr<double> vert, UniqueDevicePtr<unsigned int> facets, unsigned int numFacets) 

}
*/