#include "ShapeOptimizer.hpp"

ShapeOptimizer::ShapeOptimizer(const char * fileName)
{
    _mesh = new Mesh(fileName);
    _DMesh = new DeviceMesh(_mesh, 128);
    _gradient = new Gradient(_DMesh);

}

ShapeOptimizer::~ShapeOptimizer()
{
    delete _mesh;
    delete _DMesh;
    delete _gradient;
}
double ShapeOptimizer::gradientDesentStep(){
    _gradient->calc_force();
    _DMesh->decend_gradient(_gradient,.01);
    return _DMesh->area();
}


double ShapeOptimizer::gradientDesent(int n){ // do n gradient desent steps
    fprintf(stdout,"At Start: volume = %f \t area = %f \n",_DMesh->volume(),_DMesh->area());
    for (int i = 0; i<n; i++){
        gradientDesentStep();
        fprintf(stdout,"Step %d: volume = %f \t area = %f \n",i+1,_DMesh->volume(),_DMesh->area());
    }
    return _DMesh->volume();
}

int ShapeOptimizer::optimize(){

    return 1;

}
void ShapeOptimizer::printMesh(const char* fileName){
    Mesh optimizedMesh = _DMesh->copy_to_host();
    optimizedMesh.print(fileName);
}

