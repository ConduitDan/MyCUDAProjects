
#pragma once
#ifndef APIFactory_hpp
#define APIFactory_hpp

#include "DeviceAPI.hpp"


#ifdef __NVCC__

#include "CUDA_API.hpp"
#define API CUDA

#elif defined __OPENCL__
#include "OpenCL_API.hpp"
#define API OpenCL

#else

#error A CUDA or OpenCL compiler is required!

#endif

class APIFactory{

private:
	static DeviceAPI* myAPI;

public:
	static DeviceAPI* get_API(int blockSize);

};

#endif