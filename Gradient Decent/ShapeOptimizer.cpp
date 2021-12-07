#include "ShapeOptimizer.hpp"

ShapeOptimizer::ShapeOptimizer(const char * fileName)
{
    _mesh = new Mesh(fileName);
    _DMesh = new DeviceMesh(_mesh, 256);
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
    _DMesh->decend_gradient(_gradient,_stepSize);
    return _DMesh->area();
}


double ShapeOptimizer::gradientDesent(int n){ // do n gradient desent steps
    double volume = _DMesh->volume();
    double res = 0;
    fprintf(stdout,"At Start: volume = %f \t area = %f \n",volume,_DMesh->area());
    for (int i = 0; i<n; i++){
        int j = 0;
        gradientDesentStep();
        res = _DMesh->volume()-volume;
        while (abs(res)>tol){
            _gradient->reproject(-res);
            res = _DMesh->volume()-volume;

            j++;
            if (j>100){
                printf("Warning: Too many steps in constraint satisfaction\n");
                break;
            }
        }

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

