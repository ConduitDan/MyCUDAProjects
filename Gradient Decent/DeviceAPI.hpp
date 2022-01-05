#pragma once
#ifndef DeviceAPI_hpp
#define DeviceAPI_hpp

class DeviceAPI{
protected:
    unsigned int blockSize;

public:
    virtual void allocate(void** ptr, unsigned int size) = 0;
    virtual void copy_to_host(void * hostPointer, void * devicepointer, unsigned int size) = 0;
	virtual void copy_to_device(void* devicePointer, void* hostPointer, unsigned int size) = 0;
    virtual void deallocate(void* devicePointer) = 0;
	virtual double getGPUElement(double * vec, unsigned int index) = 0;
	virtual unsigned int getGPUElement(unsigned int * vec, unsigned int index) = 0;


    virtual double sum_of_elements(double* vec,unsigned int size,unsigned int bufferedSize) = 0;
    virtual double dotProduct(double * v1, double * v2, double * scratch, unsigned int size) = 0;
    virtual void add_with_mult(double * a,double * b, double lambda, unsigned int size) = 0;//a = a + b* lambda
    virtual void project_force(double* force,double *gradAVert,double * gradVVert, double scale,unsigned int size) = 0;
    virtual void facet_to_vertex(double* vertexValue, double* facetValue,unsigned int* vertToFacet, unsigned int* vertIndexStart,unsigned int numVert) = 0;
    virtual void area_gradient(double * gradAFacet,unsigned int* facets,double * vert,unsigned int numFacets) = 0;
    virtual void volume_gradient(double * gradVFacet,unsigned int* facets,double * vert,unsigned int numFacets) = 0;
    virtual void area(double * area, double * vert, unsigned int * facets, unsigned int numFacets) = 0;
    virtual void volume(double * volume, double * vert, unsigned int * facets, unsigned int numFacets) = 0;

    DeviceAPI(unsigned int blockSizeIn){blockSize = blockSizeIn;}

    unsigned int get_blockSize(){return blockSize;}
	void set_blockSize(unsigned int size){ blockSize = size;}


};

template <typename T>
class UniqueDevicePtr {
private:
    T* _value = nullptr;
    DeviceAPI* _myDevice;
public:
    UniqueDevicePtr(DeviceAPI* apiIn){_myDevice = apiIn;}
    UniqueDevicePtr(T* ptrIn, DeviceAPI* apiIn){
        _myDevice = apiIn;
        _value = ptrIn;
    }

    ~UniqueDevicePtr(){if(_value) _myDevice->deallocate(_value);}

    void allocate(int size){
		if (_value) _myDevice->deallocate(_value);
		_myDevice->allocate((void**) &_value, size * sizeof(T));
	}

    T* get(){return _value;}

};
#endif