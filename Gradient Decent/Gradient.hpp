#pragma once
#ifndef Gradient_hpp
#define Gradient_hpp

#include "Mesh.hpp"
#include "DeviceAPI.hpp"
#include <memory>
class DeviceMesh;

class Gradient{
protected:
	DeviceMesh *_myMesh;

	DeviceAPI *_GPU = nullptr; 

	UniqueDevicePtr<double> _gradAFacet = UniqueDevicePtr<double>(_GPU);
	UniqueDevicePtr<double> _gradAVert = UniqueDevicePtr<double>(_GPU);

	UniqueDevicePtr<double> _gradVFacet = UniqueDevicePtr<double>(_GPU);
	UniqueDevicePtr<double> _gradVVert = UniqueDevicePtr<double>(_GPU);

	UniqueDevicePtr<double> _force = UniqueDevicePtr<double>(_GPU);
	UniqueDevicePtr<double> _scratch = UniqueDevicePtr<double>(_GPU);


	void calc_gradA();
	void calc_gradV();

private:

	void calc_gradA_facet();
	void calc_gradV_facet();
	void project_to_force();



public:
	Gradient(DeviceMesh *inMesh, DeviceAPI* GPUAPIin);
	void calc_force();
	
	void reproject(double res);


	double* get_gradA(){return _gradAVert.get();}
	double* get_gradV(){return _gradVVert.get();}
	double* get_force(){return _force.get();}
};


#endif