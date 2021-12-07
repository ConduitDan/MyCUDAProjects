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

	cudaError_t _cudaStatus;


	void calc_gradA();
	void calc_gradV();
	void facet_to_vertex(double*, double*);
	void project_to_force();



public:
	Gradient(DeviceMesh *inMesh);
	~Gradient();
	void calc_force();

	double* get_force(){return _force;}
};
#endif