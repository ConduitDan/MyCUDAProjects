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
    double tol = 1e-3;
    double _stepSize = 1e-3;

    double gradientDesentStep(); // takes a gradient step

public:
    ShapeOptimizer(const char * fileName);
    ~ShapeOptimizer();

    double gradientDesent(int); // do n steps
    int optimize();
    void printMesh(const char*);
};




#endif