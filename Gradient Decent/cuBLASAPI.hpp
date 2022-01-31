#pragma once
#ifndef cuBLAS_API_hpp
#define cuBLAS_API_hpp



#include "CUDA_API.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <cusparse.h>
#include <stdio.h> 


class cuBLAS: public CUDA{

private:
    cudaError_t _cudaStatus;
	cublasStatus_t _stat;
	cublasHandle_t _handle;

	cusparseHandle_t _hSparse;
	cusparseStatus_t _sparseStat;

	// things for sparse matrix multiplcation
	bool _preProcessFacetToVert = true;
	UniqueDevicePtr<double> _bsrValA = UniqueDevicePtr<double>(this);
	UniqueDevicePtr<int> _bsrRowPtrA = UniqueDevicePtr<int>(this);
	UniqueDevicePtr<int> _bsrColIndA = UniqueDevicePtr<int>(this);
	int mb;
	int nb;
	int nnzb;
	int blockDim;

	void setup();
	//void preProcessForSparse(UniqueDevicePtr<unsigned int>* vertToFacet, UniqueDevicePtr<unsigned int>* vertIndexStart,unsigned int numVert); 

	void cuBLAS_check(const char * caller);
	void cuSPARSE_check(const char * caller);
public:
    cuBLAS();
    cuBLAS(int blocksize);

	//cublas dot()
    double dotProduct(UniqueDevicePtr<double>* v1, UniqueDevicePtr<double>* v2, UniqueDevicePtr<double>* scratch, unsigned int size);

	// cublas axpy() y= a x+y
	void add_with_mult(UniqueDevicePtr<double>* a,UniqueDevicePtr<double>* b, double lambda, unsigned int size);//a = a + b* lambda
    
	// this does z= -(x -scale y)
	// can be done with axpy() and a scal()
	void project_force(UniqueDevicePtr<double>* force,UniqueDevicePtr<double>* gradAVert,UniqueDevicePtr<double>* gradVVert, double scale,unsigned int size);
    
	// this is just a sparce matrix multipication, we can use cusparsebsrmv
	//void facet_to_vertex(UniqueDevicePtr<double>* vertexValue, UniqueDevicePtr<double>* facetValue,UniqueDevicePtr<unsigned int>* vertToFacet, UniqueDevicePtr<unsigned int>* vertIndexStart,unsigned int numVert);

};





    

#endif