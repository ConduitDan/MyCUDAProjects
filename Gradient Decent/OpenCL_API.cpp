
#include "OpenCL_API.hpp"

// TODO List
// [x] rewrite Device pointers (maybe subclass) to be able to use cl_mem
// 	What do device Pointers for openCL need to do?
//		interface can be the same (it will still need to be able to allcoate and deallocate on destuction)
//		differnece is instead of a T* it stores a cl_mem* (and thats what is given back to it)
// 		what should then happen is the API should take the full device pointer,
//		 not just the result of get (so we can still demand correct typing)
//		get should return a void* and the API should know how to cast it
// [x] fix CUDA kernals to take the whole devicePointer instead of just the raw pointer
// [x] convert CUDA kernals to openCL kernals
// [ ] write openCL api

OpenCL::OpenCL():DeviceAPI(256){
	setup();
}

OpenCL::OpenCL(int blocksize):DeviceAPI(blocksize){
	setup();
}
void OpenCL::setup(){
	int gpu = 1;
    error = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (error != CL_SUCCESS)
    {
        throw("Error: Failed to create a device group!\n");
    }
  
    // Create a compute context 
    //
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &error);
    if (!context)
    {
        throw("Error: Failed to create a compute context!\n");
    }

    // Create a command que
    //
    commands = clCreateCommandQueueWithProperties(context, device_id, 0, &error);
    if (!commands)
    {
        throw("Error: Failed to create a command commands!\n");
    }
	// register compile and register the list of kernals
	for (int i = 0; i<NUMBER_OF_PROGRAMS; i++){
		program_list[i] = clCreateProgramWithSource(context,1,&programVarNames[i],NULL,&error);
		if (!program_list[i]||error != CL_SUCCESS){
			throw("Error: Failed to create program!\n");
		}
		error = clBuildProgram(program_list[i], 0, NULL, NULL, NULL, NULL);
    	if (error != CL_SUCCESS)
		{
			size_t len;
			char buffer[2048];

			printf("Error: Failed to build program executable!\n");
			clGetProgramBuildInfo(program_list[i], device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
			printf("%s\n", buffer);
			exit(1);
		}
		kernel_list[i] = clCreateKernel(program_list[i],programNames[i], &error);
		if(!kernel_list[i] || error != CL_SUCCESS){
			throw("Error: Failed to create compute kernal!\n");

		}
		

	}




}
void OpenCL::allocate(void** ptr, unsigned int size){
	cl_mem* var = new cl_mem;
	clCreateBuffer(context,CL_MEM_READ_WRITE,size,NULL,&error);
	if (error != CL_SUCCESS)
    {
        throw("Error: Failed to allocate memory!\n");
    }

}
void OpenCL::copy_to_host(void * hostPointer, void * devicepointer, unsigned int size){
	error = clEnqueueReadBuffer(commands,*(cl_mem*) devicepointer,CL_TRUE,0,size,hostPointer,0,NULL,NULL);
	if (error != CL_SUCCESS)
    {
        throw("Error: Failed to read data to device!\n");
    }

}
void OpenCL::copy_to_device(void* devicePointer, void* hostPointer, unsigned int size){
	error = clEnqueueWriteBuffer(commands, *(cl_mem*)devicePointer, CL_TRUE,0,size,hostPointer,0,NULL,NULL);
	if (error != CL_SUCCESS)
    {
        throw("Error: Failed to write data to device!\n");
    }
	
}
void OpenCL::deallocate(void* devicePointer){
	error = clReleaseMemObject(*(cl_mem*)devicePointer);
	if (error != CL_SUCCESS) 	{
        throw("Error: Failed to deallocate memory!\n");
    }
	delete((cl_mem*)devicePointer);
}
double OpenCL::getGPUElement(double * vec, unsigned int index){
	return 0;
}
unsigned int OpenCL::getGPUElement(unsigned int * vec, unsigned int index){
	return 0;

}

void OpenCL::one_add_tree(UniqueDevicePtr<double>* vec,size_t local, size_t global){
	error =  clSetKernelArg(kernel_list[ADDTREE], 0,sizeof(cl_mem),(cl_mem*)(vec->get_void()));
	error |= clSetKernelArg(kernel_list[ADDTREE], 1,sizeof(cl_mem),(cl_mem*)(vec->get_void()));
	error |= clSetKernelArg(kernel_list[ADDTREE], 2,sizeof(double)*blockSize,NULL);
	error = clEnqueueNDRangeKernel(commands, kernel_list[ADDTREE], 1, NULL, &local, &global, 0, NULL, NULL);
	clFinish(commands);
}

double OpenCL::sum_of_elements(UniqueDevicePtr<double>* vec,unsigned int size,unsigned int bufferedSize){

	double out;

    // do the reduction each step sums blockSize*2 number of elements
	unsigned int numberOfBlocks = ceil(size / (float) blockSize / 2.0);
    size_t global = blockSize * numberOfBlocks;
	size_t local = blockSize;
	one_add_tree(vec,local,global);
    if (numberOfBlocks>1){
        for (int i = numberOfBlocks; i > 1; i /= (blockSize * 2)) {
			global /= global/(blockSize * 2);
			one_add_tree(vec,local,global);
		}
    }

    // copy the 0th element out of the vector now that it contains the sum
    copy_to_host(&out, vec->get_void(),sizeof(double));
  

    return out;

	
}
double OpenCL::dotProduct(UniqueDevicePtr<double>* v1, UniqueDevicePtr<double>* v2, UniqueDevicePtr<double>* scratch, unsigned int size){
	error =  clSetKernelArg(kernel_list[ELEMENTMULT], 0,sizeof(cl_mem),(cl_mem*)(v1->get_void()));
	error |= clSetKernelArg(kernel_list[ELEMENTMULT], 1,sizeof(cl_mem),(cl_mem*)(v2->get_void()));
	error |= clSetKernelArg(kernel_list[ELEMENTMULT], 2,sizeof(cl_mem),(cl_mem*)(scratch->get_void()));
	error |= clSetKernelArg(kernel_list[ELEMENTMULT], 3,sizeof(unsigned int),&size);

	if (error != CL_SUCCESS){
		throw("Error: Failed to set kernal args!\n");
	}
	size_t local = blockSize;
	size_t global = size;

	error = clEnqueueNDRangeKernel(commands, kernel_list[ELEMENTMULT], 1, NULL, &local, &global, 0, NULL, NULL);
	if (error != CL_SUCCESS){
		throw("Error: kernal failed!\n");
	}
	clFinish(commands);



    unsigned int bufferedSize = ceil(size/(2.0*blockSize))*2 *blockSize;
    //now sum
    double out = sum_of_elements(scratch,size, bufferedSize);
	double zero = 0;
	error = clEnqueueFillBuffer(commands,*(cl_mem*)(scratch->get_void()), &zero, sizeof(double),0,bufferedSize*sizeof(double),0,NULL,NULL);
	if (error != CL_SUCCESS){
		throw("Error: memset failed!\n");
	}
    return out;



}
void OpenCL::add_with_mult(UniqueDevicePtr<double>* a,UniqueDevicePtr<double>* b, double lambda, unsigned int size){//a = a + b* lambda
	error = clSetKernelArg(kernel_list[ADDWITHMULT], 0,sizeof(cl_mem),(cl_mem*)(a->get_void()));
	error |= clSetKernelArg(kernel_list[ADDWITHMULT], 1,sizeof(cl_mem),(cl_mem*)(b->get_void()));
	error |= clSetKernelArg(kernel_list[ADDWITHMULT], 2,sizeof(double),&lambda);
	error |= clSetKernelArg(kernel_list[ADDWITHMULT], 3,sizeof(unsigned int),&size);


		if (error != CL_SUCCESS){
		throw("Error: Failed to set kernal args!\n");
	}
	size_t local = blockSize;
	size_t global = size;

	error = clEnqueueNDRangeKernel(commands, kernel_list[ADDWITHMULT], 1, NULL, &local, &global, 0, NULL, NULL);
	clFinish(commands);
}
void OpenCL::project_force(UniqueDevicePtr<double>* force,UniqueDevicePtr<double>* gradAVert,UniqueDevicePtr<double>* gradVVert, double scale,unsigned int size){
	error =  clSetKernelArg(kernel_list[PROJECTFORCE], 0,sizeof(cl_mem),(cl_mem*)(force->get_void()));
	error |= clSetKernelArg(kernel_list[PROJECTFORCE], 1,sizeof(cl_mem),(cl_mem*)(gradAVert->get_void()));
	error |= clSetKernelArg(kernel_list[PROJECTFORCE], 2,sizeof(cl_mem),(cl_mem*)(gradVVert->get_void()));
	error |= clSetKernelArg(kernel_list[PROJECTFORCE], 3,sizeof(unsigned int),&scale);
	error |= clSetKernelArg(kernel_list[PROJECTFORCE], 4,sizeof(unsigned int),&size);

	if (error != CL_SUCCESS){
		throw("Error: Failed to set kernal args!\n");
	}
	size_t local = blockSize;
	size_t global = size;

	error = clEnqueueNDRangeKernel(commands, kernel_list[PROJECTFORCE], 1, NULL, &local, &global, 0, NULL, NULL);
	clFinish(commands);
}
void OpenCL::facet_to_vertex(UniqueDevicePtr<double>* vertexValue, UniqueDevicePtr<double>* facetValue,UniqueDevicePtr<unsigned int>* vertToFacet, UniqueDevicePtr<unsigned int>* vertIndexStart,unsigned int numVert){
	error = clSetKernelArg(kernel_list[FACETTOVERTEX], 0,sizeof(cl_mem),(cl_mem*)(vertexValue->get_void()));
	error |= clSetKernelArg(kernel_list[FACETTOVERTEX], 1,sizeof(cl_mem),(cl_mem*)(facetValue->get_void()));
	error |= clSetKernelArg(kernel_list[FACETTOVERTEX], 2,sizeof(cl_mem),(cl_mem*)(vertToFacet->get_void()));
	error |= clSetKernelArg(kernel_list[FACETTOVERTEX], 2,sizeof(cl_mem),(cl_mem*)(vertIndexStart->get_void()));

	error |= clSetKernelArg(kernel_list[FACETTOVERTEX], 3,sizeof(unsigned int),&numVert);

	if (error != CL_SUCCESS){
		throw("Error: Failed to set kernal args!\n");
	}
	size_t local = blockSize;
	size_t global = numVert;

	error = clEnqueueNDRangeKernel(commands, kernel_list[FACETTOVERTEX], 1, NULL, &local, &global, 0, NULL, NULL);
	clFinish(commands);
}

void OpenCL::area_gradient(UniqueDevicePtr<double>* gradAFacet,UniqueDevicePtr<unsigned int>* facets,UniqueDevicePtr<double>* vert,unsigned int numFacets){
	error = clSetKernelArg(kernel_list[AREAGRAD], 0,sizeof(cl_mem),(cl_mem*)(gradAFacet->get_void()));
	error |= clSetKernelArg(kernel_list[AREAGRAD], 1,sizeof(cl_mem),(cl_mem*)(facets->get_void()));
	error |= clSetKernelArg(kernel_list[AREAGRAD], 2,sizeof(cl_mem),(cl_mem*)(vert->get_void()));
	error |= clSetKernelArg(kernel_list[AREAGRAD], 3,sizeof(unsigned int),&numFacets);

	if (error != CL_SUCCESS){
		throw("Error: Failed to set kernal args!\n");
	}
	size_t local = blockSize;
	size_t global = numFacets;

	error = clEnqueueNDRangeKernel(commands, kernel_list[AREAGRAD], 1, NULL, &local, &global, 0, NULL, NULL);
	clFinish(commands);

}
void OpenCL::volume_gradient(UniqueDevicePtr<double>* gradVFacet,UniqueDevicePtr<unsigned int>* facets,UniqueDevicePtr<double>* vert,unsigned int numFacets){
	error = clSetKernelArg(kernel_list[VOLUMEGRAD], 0,sizeof(cl_mem),(cl_mem*)(gradVFacet->get_void()));
	error |= clSetKernelArg(kernel_list[VOLUMEGRAD], 1,sizeof(cl_mem),(cl_mem*)(facets->get_void()));
	error |= clSetKernelArg(kernel_list[VOLUMEGRAD], 2,sizeof(cl_mem),(cl_mem*)(vert->get_void()));
	error |= clSetKernelArg(kernel_list[VOLUMEGRAD], 3,sizeof(unsigned int),&numFacets);

	if (error != CL_SUCCESS){
		throw("Error: Failed to set kernal args!\n");
	}
	size_t local = blockSize;
	size_t global = numFacets;

	error = clEnqueueNDRangeKernel(commands, kernel_list[VOLUMEGRAD], 1, NULL, &local, &global, 0, NULL, NULL);
	clFinish(commands);

}

void OpenCL::area(UniqueDevicePtr<double>* area, UniqueDevicePtr<double>* vert, UniqueDevicePtr<unsigned int>* facets, const unsigned int numFacets){

	error = clSetKernelArg(kernel_list[AREA], 0,sizeof(cl_mem),(cl_mem*)(area->get_void()));
	error |= clSetKernelArg(kernel_list[AREA], 1,sizeof(cl_mem),(cl_mem*)(vert->get_void()));
	error |= clSetKernelArg(kernel_list[AREA], 2,sizeof(cl_mem),(cl_mem*)(facets->get_void()));
	error |= clSetKernelArg(kernel_list[AREA], 3,sizeof(unsigned int),&numFacets);

	if (error != CL_SUCCESS){
		throw("Error: Failed to set kernal args!\n");
	}
	size_t local = blockSize;
	size_t global = numFacets;

	error = clEnqueueNDRangeKernel(commands, kernel_list[AREA], 1, NULL, &local, &global, 0, NULL, NULL);
	clFinish(commands);

}
void OpenCL::volume(UniqueDevicePtr<double>* volume, UniqueDevicePtr<double>* vert, UniqueDevicePtr<unsigned int>* facets, unsigned int numFacets){

	error = clSetKernelArg(kernel_list[VOLUME], 0,sizeof(cl_mem),(cl_mem*)(volume->get_void()));
	error |= clSetKernelArg(kernel_list[VOLUME], 1,sizeof(cl_mem),(cl_mem*)(vert->get_void()));
	error |= clSetKernelArg(kernel_list[VOLUME], 2,sizeof(cl_mem),(cl_mem*)(facets->get_void()));
	error |= clSetKernelArg(kernel_list[VOLUME], 3,sizeof(unsigned int),&numFacets);

	if (error != CL_SUCCESS){
		throw("Error: Failed to set kernal args!\n");
	}
	size_t local = blockSize;
	size_t global = numFacets;

	error = clEnqueueNDRangeKernel(commands, kernel_list[VOLUME], 1, NULL, &local, &global, 0, NULL, NULL);
	clFinish(commands);

}

