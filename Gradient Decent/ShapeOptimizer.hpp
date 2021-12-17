#pragma once
#ifndef ShapeOptimizer_hpp
#define ShapeOptimizer_hpp

#include "Mesh.hpp"
#include "Gradient.hpp"

class ShapeOptimizer
{
private:
    Mesh* _mesh;
    DeviceMesh* _DMesh;
    Gradient* _gradient;
    double tol = 1e-10;
    double _startingVol = 0;
    double _stepSize = 0.1;
    double _dAtol = 1e-16;

    int _maxConstraintSteps = 20;

    double gradientDesentStep(); // takes a gradient step
    double reproject_constraints();

public:
    ShapeOptimizer(const char * fileName);
    ~ShapeOptimizer();

    double gradientDesent(int); // do n steps
    int optimize();
    void printMesh(const char*);
    Mesh get_optimized_mesh(){return _DMesh->copy_to_host();}
};




#endif