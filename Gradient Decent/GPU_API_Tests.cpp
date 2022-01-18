//CUDATests

//#include "Mesh.hpp"
#include "ShapeOptimizer.hpp"
#include <iostream>
#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>
#include "APIFactory.hpp"


// THings aren't working because the unique device pointers are not being passed by reference





BOOST_AUTO_TEST_CASE(Device_Pointers){
	//copy an array the the device and copy it back 
	DeviceAPI* GPU = APIFactory::get_API(128);
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


BOOST_AUTO_TEST_CASE(Device_Pointers_2){
	//copy an array the the device and copy it back 
	DeviceAPI* GPU = APIFactory::get_API(128);

	double myArray[5] = {1,2,3,4,5};

	UniqueDevicePtr<double>devArray(GPU);

	devArray.allocate(5);
	GPU->copy_to_device(devArray.get(),myArray,5*sizeof(double));

	double myArrayCopiedBack[5];

	GPU->copy_to_host(myArrayCopiedBack,devArray.get(),5*sizeof(double));

	for (int i = 0; i<5; i++){
		BOOST_CHECK(myArrayCopiedBack[i]==myArray[i]);
		if (myArrayCopiedBack[i]!=myArray[i]){
			printf("%f (actual) != %f (expected)\n",myArrayCopiedBack[i],myArray[i]);
		}
	}
}


BOOST_AUTO_TEST_CASE(add_with_mult){
	
	DeviceAPI* GPU = APIFactory::get_API(128);

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
	DeviceAPI* GPU = APIFactory::get_API(128);
	double myArray[5] = {1,2,3,4,5};

	UniqueDevicePtr<double>* vec = new UniqueDevicePtr<double>(GPU);
	vec->allocate(128*2);
	
	GPU->copy_to_device(vec->get(),myArray,5*sizeof(double));
	int size = 5;
	int bufferedSize = 128*2;

	double ans = GPU->sum_of_elements(vec, size,bufferedSize);
	delete vec;
	BOOST_CHECK(ans==1+2+3+4+5);


}


BOOST_AUTO_TEST_CASE(sum_of_elements2){
	DeviceAPI* GPU = APIFactory::get_API(128);
	double myArray[5] = {1,2,3,4,5};

	UniqueDevicePtr<double> vec(GPU);
	vec.allocate(128*2);
	
	GPU->copy_to_device(vec.get(),myArray,5*sizeof(double));
	int size = 5;
	int bufferedSize = 128*2;
	double ans = GPU->sum_of_elements(&vec, size,bufferedSize);
	BOOST_CHECK(ans==1+2+3+4+5);


}



BOOST_AUTO_TEST_CASE(dotProduct){
	DeviceAPI* GPU = APIFactory::get_API(128);
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

	DeviceAPI* GPU = APIFactory::get_API(128);


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
		printf("expected %f, got %f\n",expected[i],actual[i]);
		BOOST_CHECK(expected[i]==actual[i]);
	}

}
/*

BOOST_AUTO_TEST_CASE(facet_to_vertex){
	facet_to_vertex(UniqueDevicePtr<double> vertexValue, UniqueDevicePtr<double> facetValue,UniqueDevicePtr<unsigned int> vertToFacet, UniqueDevicePtr<unsigned int> vertIndexStart,unsigned int numVert)

}
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