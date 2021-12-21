#pragma once
#ifndef Gradient_hpp
#define Gradient_hpp

#include "Mesh.hpp"
#include <memory>
class DeviceMesh;

class Gradient{
protected:
	DeviceMesh *_myMesh;

	DeviceAPI * GPU = CUDA::Instance(); 

	using unique_device_ptr = std::unique_ptr<double,decltype(&(GPU->deallocate))>;
	unique_device_ptr _gradAFacet;
	unique_device_ptr _gradAVert;

	unique_device_ptr _gradVFacet;
	unique_device_ptr _gradVVert;

	unique_device_ptr _force;
	unique_device_ptr _scratch;




	void calc_gradA();
	void calc_gradV();

private:

	void calc_gradA_facet();
	void calc_gradV_facet();
	void project_to_force();



public:
	Gradient(DeviceMesh *inMesh);
	void calc_force();
	
	void reproject(double res);


	double* get_gradA(){return _gradAVert.get();}
	double* get_gradV(){return _gradVVert.get();}
	double* get_force(){return _force.get();}
};


#endif