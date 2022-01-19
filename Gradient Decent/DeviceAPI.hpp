#pragma once
#ifndef DeviceAPI_hpp
#define DeviceAPI_hpp


template <typename T> class UniqueDevicePtr;




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


    virtual double sum_of_elements(UniqueDevicePtr<double>* vec,unsigned int size,unsigned int bufferedSize) = 0;
    virtual double dotProduct(UniqueDevicePtr<double>* v1, UniqueDevicePtr<double>* v2, UniqueDevicePtr<double>* scratch, unsigned int size) = 0;
    virtual void add_with_mult(UniqueDevicePtr<double>* a,UniqueDevicePtr<double>* b, double lambda, unsigned int size) = 0;//a = a + b* lambda
    virtual void project_force(UniqueDevicePtr<double>* force,UniqueDevicePtr<double>* gradAVert,UniqueDevicePtr<double>* gradVVert, double scale,unsigned int size) = 0;
    virtual void facet_to_vertex(UniqueDevicePtr<double>* vertexValue, UniqueDevicePtr<double>* facetValue,UniqueDevicePtr<unsigned int>* vertToFacet, UniqueDevicePtr<unsigned int>* vertIndexStart,unsigned int numVert) = 0;
    virtual void area_gradient(UniqueDevicePtr<double>* gradAFacet,UniqueDevicePtr<unsigned int>* facets,UniqueDevicePtr<double>* vert,unsigned int numFacets) = 0;
    virtual void volume_gradient(UniqueDevicePtr<double>* gradVFacet,UniqueDevicePtr<unsigned int>* facets,UniqueDevicePtr<double>* vert,unsigned int numFacets) = 0;
    virtual void area(UniqueDevicePtr<double>* area, UniqueDevicePtr<double>* vert, UniqueDevicePtr<unsigned int>* facets, unsigned int numFacets) = 0;
    virtual void volume(UniqueDevicePtr<double>* volume, UniqueDevicePtr<double>* vert, UniqueDevicePtr<unsigned int>* facets, unsigned int numFacets) = 0;



    DeviceAPI(unsigned int blockSizeIn){blockSize = blockSizeIn;}

    unsigned int get_blockSize(){return blockSize;}
	void set_blockSize(unsigned int size){ blockSize = size;}


};

template <typename T>
class UniqueDevicePtr {
private:
    void* _value = nullptr;
    DeviceAPI* _myDevice;
public:
    UniqueDevicePtr(DeviceAPI* apiIn){_myDevice = apiIn;}
    UniqueDevicePtr(void* ptrIn, DeviceAPI* apiIn){
        _myDevice = apiIn;
        _value = ptrIn;
    }

    ~UniqueDevicePtr(){if(_value) _myDevice->deallocate(_value);}

    void allocate(int size){
		//if (_value) _myDevice->deallocate(_value);
		_myDevice->allocate((void**) &_value, size * sizeof(T));
	}

    void* get_void(){return _value;}
	T* get(){return (T*) _value;}


};
#endif