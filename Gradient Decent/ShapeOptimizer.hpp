#pragma once
#ifndef ShapeOptimizer_hpp
#define ShapeOptimizer_hpp

#include "Mesh.hpp"
class ShapeOptimizer
{
private:
    Mesh* myMesh;
    DeviceMesh* myDMesh;
    Gradient* myGradient;
    double tol = 1e-8;

public:
    ShapeOptimizer(const char * fileName);
    ~ShapeOptimizer();

    optimize()
};

ShapeOptimizer::ShapeOptimizer(const char * fileName)
{
    myMesh = new Mesh(fileName);
    myDMesh = new DeviceMesh(myMesh, 128);
    myGradient = new Gradient(myDMesh);

}

ShapeOptimizer::~ShapeOptimizer()
{
    delete myMesh;
    delete myDMesh;
    delete myGradient;
}




#endif