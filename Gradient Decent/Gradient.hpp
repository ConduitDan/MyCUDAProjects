#pragma once
#ifndef Gradient_hpp
#define Gradient_hpp

#include "Mesh.hpp"
class DeviceMesh;

class Gradient{
private:
	DeviceMesh *_myMesh;

	double *_gradAFacet = nullptr;
	double *_gradAVert = nullptr;

	double *_gradVFacet = nullptr;
	double *_gradVVert = nullptr;

	double *_force = nullptr;
	double *_scratch = nullptr;

	cudaError_t _cudaStatus = cudaSetDevice(0);


	void calc_gradA();
	void calc_gradV();
	void facet_to_vertex(double*, double*);
	void project_to_force();



public:
	Gradient(DeviceMesh *inMesh);
	~Gradient();
	void calc_force();
	void reproject(double res);


	double* get_gradA(){return _gradAVert;}
	double* get_gradV(){return _gradVVert;}
	double* get_force(){return _force;}
};
#endif