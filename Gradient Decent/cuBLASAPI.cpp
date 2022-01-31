#include "cuBLASAPI.hpp"

cuBLAS::cuBLAS():CUDA(){
	setup();
}
cuBLAS::cuBLAS(int blockSize):CUDA(blockSize){
	setup();
}
void cuBLAS::setup(){
	_stat = cublasCreate(&_handle);
	cuBLAS_check("setup");
	printf("using cuBLAS\n");

	_sparseStat = cusparseCreate(&_hSparse);
}


//cublas dot()
double cuBLAS::dotProduct(UniqueDevicePtr<double>* v1, UniqueDevicePtr<double>* v2, UniqueDevicePtr<double>* scratch, unsigned int size){
	double result;

	
	_stat =  cublasDdot (_handle, size,\
                           (const double *)v1->get(), 1,\
                           (const double *)v2->get(), 1,\
                           &result);
	cuBLAS_check("dot product");

	return result;
}

// cublas axpy() y= a x+y
void cuBLAS::add_with_mult(UniqueDevicePtr<double>* a,UniqueDevicePtr<double>* b, double lambda, unsigned int size){//a = a + b* lambda
	_stat = cublasDaxpy(_handle, size,\
                           (const double*) &lambda,\
                           (const double*) b->get(), 1,\
                           (double *) a->get(), 1);
	cuBLAS_check("add with mult");


}

// this does z= -(x -scale y)
// can be done with axpy() and a scal()
void cuBLAS::project_force(UniqueDevicePtr<double>* force,UniqueDevicePtr<double>* gradAVert,UniqueDevicePtr<double>* gradVVert, double scale,unsigned int size){
		//first copy gradA to the force vectory
		_stat = cublasDcopy(_handle, size,\
                           (const double *) gradAVert->get(), 1,\
                           (double *) force->get(), 1);
		cuBLAS_check("copy");

		// now do (x-scale y)
		scale = -scale;
		_stat = cublasDaxpy(_handle, size,\
                           (const double*) &scale,\
                           (const double*) gradVVert->get(), 1,\
                           (double *) force->get(), 1);
		cuBLAS_check("add and mult");
		// might just be faster to not do this scale here and have the force just be negitive
		double alpha = -1;
		_stat = cublasDscal(_handle, size,\
                            &alpha,\
                            (double *) force->get(), 1);
		cuBLAS_check("scale");
}
/*
// this is just a sparce matrix multipication, we can use cusparsebsrmv
void cuBLAS::facet_to_vertex(UniqueDevicePtr<double>* vertexValue, UniqueDevicePtr<double>* facetValue,UniqueDevicePtr<unsigned int>* vertToFacet, UniqueDevicePtr<unsigned int>* vertIndexStart,unsigned int numVert){
	//vert_i = A_{i,j} Facet_j

	// we keep track of this is CSR format where facetValue is the csrColInd vertToFacet and vertIndexStart is the csrRowPtr
	// the values are assumed to be 1
	 
	// to prep for this we acutally need to make this 3x as long beacsue there are x,y, and z components

	if (_preProcessFacetToVert){
		preProcessForSparse(vertToFacet, vertIndexStart, numVert);
	}
	cusparseMatDescr_t descr;
	cusparseCreateMatDescr(&descr);
	// computes y = alpha A*x + beta * y
	double alpha = 1;
	double beta = 0;
	//prinf("attempting sprase matrix multiplication\n");
	_sparseStat = cusparseDbsrmv(_hSparse,
               CUSPARSE_DIRECTION_ROW,\
               CUSPARSE_OPERATION_NON_TRANSPOSE,\
               mb,\
               nb,\
               nnzb,\
               &alpha,\
               descr,\
               (const double*) _bsrValA.get(),\
               (const int*) _bsrRowPtrA.get(),\
               (const int*) _bsrColIndA.get(),\
               blockDim,\
               (const double*) facetValue->get(),\
               &beta,\
               (double*) vertexValue->get());

	cuSPARSE_check("Dbsrmv, the sparse matrix mult");
	//prinf("sparse matrix multipacaltion compete\n");

}


void cuBLAS::preProcessForSparse(UniqueDevicePtr<unsigned int>* vertToFacet, UniqueDevicePtr<unsigned int>* vertIndexStart,unsigned int numVert){
	//prinf("starting sparce setup\n");
	UniqueDevicePtr<double> csrValA = UniqueDevicePtr<double>(this);
	UniqueDevicePtr<int> csrRowPtrA = UniqueDevicePtr<int>(this);
	UniqueDevicePtr<int> csrColIndA = UniqueDevicePtr<int>(this);


	// need to create numFacet*3 values (this should be the last number in vertIndexStart) 
	int numFacet = getGPUElement(vertIndexStart->get(),numVert)/3;
	//prinf("found %d faces\n",numFacet);
	// first create the values
	double * hostVals = new double[numFacet*9];
	for (int i = 0; i<numFacet*9; i++){hostVals[i] = 1;};

	csrValA.allocate(numFacet*9);
	copy_to_device(csrValA.get(), hostVals, numFacet*9*sizeof(double));
	//prinf("coppied vals to device\n");


	unsigned int * hostV2F = new unsigned int[numFacet*3];
	unsigned int * hostVIS = new unsigned int[numVert+1];
	
	copy_to_host(hostV2F,vertToFacet->get(),numFacet*3*sizeof(unsigned int));

	//prinf("VtF copied to host\n");

	copy_to_host(hostVIS,vertIndexStart->get(),(numVert+1)*sizeof(unsigned int));
	//prinf("FtV start copied to host\n");

	int * hostcsrColIndA = new int[numFacet*9];
	int * hostcsrRowPtrA = new int[numVert*3+1];

	int numEle= 0;
	for (int i = 0; i<numVert; i++){
		// first find the diff for this vertex i.e the number of elements it has
		numEle = hostVIS[i+1]-hostVIS[i];
		hostcsrRowPtrA[i*3] = hostVIS[i]*3;
		hostcsrRowPtrA[i*3+1] = hostVIS[i]*3+numEle;
		hostcsrRowPtrA[i*3+2] = hostVIS[i]*3+numEle*2;
	}
	hostcsrRowPtrA[numVert*3] = numFacet*9;
	for (int i = 0; i<numFacet*3; i++){
		hostcsrColIndA[i*3] = hostV2F[i]*3;
		hostcsrColIndA[i*3+1] = hostV2F[i]*3+1;
		hostcsrColIndA[i*3+2] = hostV2F[i]*3+2;
	}

	csrRowPtrA.allocate(numVert*3+1);
	csrColIndA.allocate(numFacet*9);

	copy_to_device(csrRowPtrA.get(),hostcsrRowPtrA,(numVert*3+1)*sizeof(int));
	copy_to_device(csrColIndA.get(),hostcsrColIndA,numFacet*9*sizeof(int));
	//prinf("crs description compelete\n");

	// ok  now we have a csr Sparse matrix set up
	// we now need to move this over to bsr for the matrix multiplication
	
	// this is an nxm matrix where m is 3*vert and n = 9*facet
	// the number of non zero elements nnz is 9*facet (each facet has 3 verteices each is a 3 vector)

	int m = 3*numVert;
	int n = 9*numFacet;
	blockDim = 3; //lets try 3? its all divisable by 3;


	int base;
	mb = (m + blockDim-1)/blockDim;
	nb = (n + blockDim-1)/blockDim;
	_bsrRowPtrA.allocate((mb+1));



	// nnzTotalDevHostPtr points to host memory
	int *nnzTotalDevHostPtr = &nnzb;

	cusparseMatDescr_t descrA;
	cusparseCreateMatDescr(&descrA);
	//prinf("attempting phase 1 of crs -> brs conversion\n");
	_sparseStat = cusparseXcsr2bsrNnz(_hSparse,CUSPARSE_DIRECTION_ROW,m,n,\
						descrA,csrRowPtrA.get(),csrColIndA.get(),\
						blockDim,\
						descrA,_bsrRowPtrA.get(),nnzTotalDevHostPtr);
	cuSPARSE_check("csr2bsrNNZ");
	//prinf("phase 1 of crs -> brs conversion complete\n");



	if (NULL != nnzTotalDevHostPtr){
    	nnzb = *nnzTotalDevHostPtr;
	} else{
    	cudaMemcpy(&nnzb, _bsrRowPtrA.get()+mb, sizeof(int), cudaMemcpyDeviceToHost);
    	cudaMemcpy(&base, _bsrRowPtrA.get(), sizeof(int), cudaMemcpyDeviceToHost);
    	nnzb -= base;
	}

	_bsrColIndA.allocate(nnzb);
	_bsrValA.allocate((blockDim*blockDim)*nnzb);
	
	_sparseStat = cusparseDcsr2bsr(_hSparse,
					CUSPARSE_DIRECTION_ROW,
					m,
					n,
					descrA,
					(const double*) csrValA.get(),
					(const int*) csrRowPtrA.get(),
					(const int*) csrColIndA.get(),
					blockDim,
					descrA,
					(double*) _bsrValA.get(),
					(int*) _bsrRowPtrA.get(),
					(int*) _bsrColIndA.get());
					
	cuSPARSE_check("csr2bsr");
	//prinf("phase 2 of crs -> brs conversion complete\n");

	delete[](hostVals);
	delete[](hostcsrColIndA);
	delete[](hostcsrRowPtrA);
	//prinf("crs -> brs conversion complete\n");
}
*/
void cuBLAS::cuBLAS_check(const char * caller){
	if (_stat != CUBLAS_STATUS_SUCCESS) {
		printf("cuBLAS error from %s",caller);
		throw("cuBLAS error");
    }
	cuda_sync_and_check(caller);

}
void cuBLAS::cuSPARSE_check(const char * caller){
	if (_sparseStat != CUSPARSE_STATUS_SUCCESS) {
		printf("cuBLAS error from %s",caller);
		throw("cuBLAS error");
    }
		cuda_sync_and_check(caller);


}