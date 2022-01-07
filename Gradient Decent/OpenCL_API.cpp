
#include "OpenCL_API.hpp"

// TODO List
// [ ] rewrite Device pointers (maybe subclass) to be able to use cl_mem
// 	What do device Pointers for openCL need to do?
//		interface can be the same (it will still need to be able to allcoate and deallocate on destuction)
//		differnece is instead of a T* it stores a cl_mem* (and thats what is given back to it)
// 		what should then happen is the API should take the full device pointer,
//		 not just the result of get (so we can still demand correct typing)
//		get should return a void* and the API should know how to cast it
// [ ] fix CUDA kernals to take the whole devicePointer instead of just the raw pointer
// [ ] convert CUDA kernals to openCL kernals
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
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }
  
    // Create a compute context 
    //
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    // Create a command que
    //
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }
}
void allocate(void** ptr, unsigned int size){

}
    void copy_to_host(void * hostPointer, void * devicepointer, unsigned int size);
    void copy_to_device(void* devicePointer, void* hostPointer, unsigned int size);
    void deallocate(void* devicePointer);
	double getGPUElement(double * vec, unsigned int index);
	unsigned int getGPUElement(unsigned int * vec, unsigned int index);