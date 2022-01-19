
#include "OpenCL_API.hpp"
#define MAX_SOURCE_SIZE (0x100000)
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
// [x] write openCL api
// find out why context is going bad
// kernal-> kernel
// write the rest of the tests
OpenCL::OpenCL():DeviceAPI(256){
	setup();
}

OpenCL::OpenCL(int blocksize):DeviceAPI(blocksize){
	setup();
}
void OpenCL::setup(){
	cl_uint num_entries = 2;
	cl_platform_id *platforms;
	cl_uint *num_platforms;

	char retvalue[50];


	size_t kernel_code_size;
	cl_uint ret_num_platforms;
	cl_platform_id platform_id = NULL;
	cl_uint ret_num_devices;
	char buildLog[1024];
	int i;
	cl_device_id OPENCL_DEVICE_ID;

	error = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	checkSuccess("clGetPlatformIDs");
	printf("--->available number of platforms = %d\n", ret_num_platforms);
	error = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
	checkSuccess("clGetDeviceIDs");
	printf("--->available number of devices = %d\n", ret_num_devices);
    
  
	// Create a compute context 
	//
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &error);
	checkSuccess("create context");
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
	checkSuccess("create command queue");


    FILE *fp = fopen(sourcepath, "r");
    if (!fp) {
	fprintf(stderr, "Failed to open kernel file '%s'\n", sourcepath);
    }
	
    char *source_str = (char*)malloc(MAX_SOURCE_SIZE);
    size_t source_size= fread( source_str, 1, MAX_SOURCE_SIZE, fp);


	program = clCreateProgramWithSource(context,1,(const char**)&source_str,&source_size,&error);
	if (!program||error != CL_SUCCESS){
		printf("failed to create program!\n");
		checkSuccess("create program");
	}
	error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (error != CL_SUCCESS)
	{
		size_t len;
		char buffer[20480];

		printf("Error: Failed to build program executable! %s\n",getErrorString(error));

		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}



	// register compile and register the list of kernals
	for (int i = 0; i<NUMBER_OF_PROGRAMS; i++){	
		kernel_list[i] = clCreateKernel(program,programNames[i], &error);
		if(!kernel_list[i] || error != CL_SUCCESS){
			throw("Error: Failed to create compute kernal %s!\n",programNames[i]);

		}
		

	}




}
void OpenCL::allocate(void** ptr, unsigned int size){
	cl_mem* var = new cl_mem;
	*var = clCreateBuffer(context,CL_MEM_READ_WRITE,size,NULL,&error);
	checkSuccess("allocate");
	*ptr = (void *)var;

}
void OpenCL::copy_to_host(void * hostPointer, void * devicepointer, unsigned int size){
	error = clEnqueueReadBuffer(commands,*(cl_mem*) devicepointer,CL_TRUE,0,size,hostPointer,0,NULL,NULL);
	checkSuccess("read data from device");
}
void OpenCL::copy_to_device(void* devicePointer, void* hostPointer, unsigned int size){
	error = clEnqueueWriteBuffer(commands, *(cl_mem*)devicePointer, CL_TRUE,0,size,hostPointer,0,NULL,NULL);
	checkSuccess("write data to device");	
}
void OpenCL::deallocate(void* devicePointer){
	error = clReleaseMemObject(*(cl_mem*)devicePointer);
	checkSuccess("deallocate");
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
	checkSuccess("set kernal args add tree");

	error = clEnqueueNDRangeKernel(commands, kernel_list[ADDTREE], 1, NULL, &local, &global, 0, NULL, NULL);
	clFinish(commands);
	checkSuccess("perform addTreeStep");

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
	checkSuccess("set kernal args dot product");

	size_t local = blockSize;

	size_t global;
	if (size<blockSize){
		global = blockSize;
	} else {global = size;}

	error = clEnqueueNDRangeKernel(commands, kernel_list[ELEMENTMULT], 1, NULL, &local, &global, 0, NULL, NULL);
	checkSuccess("perform dot product");

	clFinish(commands);



    unsigned int bufferedSize = ceil(size/(2.0*blockSize))*2 *blockSize;
    //now sum
    double out = sum_of_elements(scratch,size, bufferedSize);
	double zero = 0;
	error = clEnqueueFillBuffer(commands,*(cl_mem*)(scratch->get_void()), &zero, sizeof(double),0,bufferedSize*sizeof(double),0,NULL,NULL);
	checkSuccess("memset");

    return out;



}
void OpenCL::add_with_mult(UniqueDevicePtr<double>* a,UniqueDevicePtr<double>* b, double lambda, unsigned int size){//a = a + b* lambda
	error = clSetKernelArg(kernel_list[ADDWITHMULT], 0,sizeof(cl_mem),(cl_mem*)(a->get_void()));
	error |= clSetKernelArg(kernel_list[ADDWITHMULT], 1,sizeof(cl_mem),(cl_mem*)(b->get_void()));
	error |= clSetKernelArg(kernel_list[ADDWITHMULT], 2,sizeof(double),&lambda);
	error |= clSetKernelArg(kernel_list[ADDWITHMULT], 3,sizeof(unsigned int),&size);
	checkSuccess("set kernal args add_with_mult");

	size_t local = blockSize;

	size_t global;
	if (size<blockSize){
		global = blockSize;
	} else {global = size;}

	error = clEnqueueNDRangeKernel(commands, kernel_list[ADDWITHMULT], 1, NULL, &local, &global, 0, NULL, NULL);
	checkSuccess("perform a+b*c");
	clFinish(commands);
}
void OpenCL::project_force(UniqueDevicePtr<double>* force,UniqueDevicePtr<double>* gradAVert,UniqueDevicePtr<double>* gradVVert, double scale,unsigned int size){
	error =  clSetKernelArg(kernel_list[PROJECTFORCE], 0,sizeof(cl_mem),(cl_mem*)(force->get_void()));
	error |= clSetKernelArg(kernel_list[PROJECTFORCE], 1,sizeof(cl_mem),(cl_mem*)(gradAVert->get_void()));
	error |= clSetKernelArg(kernel_list[PROJECTFORCE], 2,sizeof(cl_mem),(cl_mem*)(gradVVert->get_void()));
	error |= clSetKernelArg(kernel_list[PROJECTFORCE], 3,sizeof(double),&scale);
	error |= clSetKernelArg(kernel_list[PROJECTFORCE], 4,sizeof(unsigned int),&size);
	checkSuccess("set kernel args project vector");
	size_t local = blockSize;
	size_t global = size;
	if (global<local){
		global = local;
	}

	error = clEnqueueNDRangeKernel(commands, kernel_list[PROJECTFORCE], 1, NULL, &local, &global, 0, NULL, NULL);
	checkSuccess("project vector");
	clFinish(commands);
}
void OpenCL::facet_to_vertex(UniqueDevicePtr<double>* vertexValue, UniqueDevicePtr<double>* facetValue,UniqueDevicePtr<unsigned int>* vertToFacet, UniqueDevicePtr<unsigned int>* vertIndexStart,unsigned int numVert){
	error = clSetKernelArg(kernel_list[FACETTOVERTEX], 0,sizeof(cl_mem),(cl_mem*)(vertexValue->get_void()));
	error |= clSetKernelArg(kernel_list[FACETTOVERTEX], 1,sizeof(cl_mem),(cl_mem*)(facetValue->get_void()));
	error |= clSetKernelArg(kernel_list[FACETTOVERTEX], 2,sizeof(cl_mem),(cl_mem*)(vertToFacet->get_void()));
	error |= clSetKernelArg(kernel_list[FACETTOVERTEX], 3,sizeof(cl_mem),(cl_mem*)(vertIndexStart->get_void()));
	error |= clSetKernelArg(kernel_list[FACETTOVERTEX], 4,sizeof(unsigned int),&numVert);

	checkSuccess("set kernal args facet to vertex");
	size_t local = blockSize;

		size_t global;
	if (numVert<blockSize){
		global = blockSize;
	} else {global = numVert;}


	error = clEnqueueNDRangeKernel(commands, kernel_list[FACETTOVERTEX], 1, NULL, &local, &global, 0, NULL, NULL);
	checkSuccess("convert from facet to vertex");
	clFinish(commands);
}

void OpenCL::area_gradient(UniqueDevicePtr<double>* gradAFacet,UniqueDevicePtr<unsigned int>* facets,UniqueDevicePtr<double>* vert,unsigned int numFacets){
	error = clSetKernelArg(kernel_list[AREAGRAD], 0,sizeof(cl_mem),(cl_mem*)(gradAFacet->get_void()));
	error |= clSetKernelArg(kernel_list[AREAGRAD], 1,sizeof(cl_mem),(cl_mem*)(facets->get_void()));
	error |= clSetKernelArg(kernel_list[AREAGRAD], 2,sizeof(cl_mem),(cl_mem*)(vert->get_void()));
	error |= clSetKernelArg(kernel_list[AREAGRAD], 3,sizeof(unsigned int),&numFacets);

	checkSuccess("set kernal args area grad");
	size_t local = blockSize;
	size_t global = numFacets;
	if (global<local){
		global = local;
	}


	error = clEnqueueNDRangeKernel(commands, kernel_list[AREAGRAD], 1, NULL, &local, &global, 0, NULL, NULL);
	checkSuccess("calculate area gradient");
	clFinish(commands);

}
void OpenCL::volume_gradient(UniqueDevicePtr<double>* gradVFacet,UniqueDevicePtr<unsigned int>* facets,UniqueDevicePtr<double>* vert,unsigned int numFacets){
	error = clSetKernelArg(kernel_list[VOLUMEGRAD], 0,sizeof(cl_mem),(cl_mem*)(gradVFacet->get_void()));
	error |= clSetKernelArg(kernel_list[VOLUMEGRAD], 1,sizeof(cl_mem),(cl_mem*)(facets->get_void()));
	error |= clSetKernelArg(kernel_list[VOLUMEGRAD], 2,sizeof(cl_mem),(cl_mem*)(vert->get_void()));
	error |= clSetKernelArg(kernel_list[VOLUMEGRAD], 3,sizeof(unsigned int),&numFacets);

	checkSuccess("set kernal args volume grad");
	size_t local = blockSize;
	size_t global = numFacets;
	if (global<local){
		global = local;
	}

	error = clEnqueueNDRangeKernel(commands, kernel_list[VOLUMEGRAD], 1, NULL, &local, &global, 0, NULL, NULL);
	checkSuccess("calculate volume gradeint");
	clFinish(commands);

}

void OpenCL::area(UniqueDevicePtr<double>* area, UniqueDevicePtr<double>* vert, UniqueDevicePtr<unsigned int>* facets, const unsigned int numFacets){

	error = clSetKernelArg(kernel_list[AREA], 0,sizeof(cl_mem),(cl_mem*)(area->get_void()));
	error |= clSetKernelArg(kernel_list[AREA], 1,sizeof(cl_mem),(cl_mem*)(vert->get_void()));
	error |= clSetKernelArg(kernel_list[AREA], 2,sizeof(cl_mem),(cl_mem*)(facets->get_void()));
	error |= clSetKernelArg(kernel_list[AREA], 3,sizeof(unsigned int),&numFacets);

	checkSuccess("set kernal args area");
	size_t local = blockSize;
	size_t global = numFacets;
	if (global<local){
		global = local;
	}

	error = clEnqueueNDRangeKernel(commands, kernel_list[AREA], 1, NULL, &local, &global, 0, NULL, NULL);
	checkSuccess("calculate area");
	clFinish(commands);

}
void OpenCL::volume(UniqueDevicePtr<double>* volume, UniqueDevicePtr<double>* vert, UniqueDevicePtr<unsigned int>* facets, unsigned int numFacets){

	error = clSetKernelArg(kernel_list[VOLUME], 0,sizeof(cl_mem),(cl_mem*)(volume->get_void()));
	error |= clSetKernelArg(kernel_list[VOLUME], 1,sizeof(cl_mem),(cl_mem*)(vert->get_void()));
	error |= clSetKernelArg(kernel_list[VOLUME], 2,sizeof(cl_mem),(cl_mem*)(facets->get_void()));
	error |= clSetKernelArg(kernel_list[VOLUME], 3,sizeof(unsigned int),&numFacets);

	checkSuccess("set kernal args volume");
	size_t local = blockSize;
	size_t global = numFacets;
	if (global<local){
		global = local;
	}

	error = clEnqueueNDRangeKernel(commands, kernel_list[VOLUME], 1, NULL, &local, &global, 0, NULL, NULL);
	checkSuccess("calculate volume");
	clFinish(commands);

}
void OpenCL::checkSuccess(const char* caller){
	if (error != CL_SUCCESS)
    {
		printf("%s from %s\n",getErrorString(error),caller);
        throw("Error: Failed to %s!\n",caller);
    }
}

const char* OpenCL::getErrorString(cl_int error)
{
switch(error){
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}