#include "ShapeOptimizer.hpp"

ShapeOptimizer::ShapeOptimizer(const char * fileName)
{
    _GPU = APIFactory::get_API(256);
    _mesh = new Mesh(fileName);
    _DMesh = new DeviceMesh(_mesh,_GPU);
    _gradient = new Gradient(_DMesh,_GPU);
    _startingVol = _DMesh->volume();

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
    reproject_constraints();
    return _DMesh->area();
}

double ShapeOptimizer::reproject_constraints(){
    double res = _startingVol -_DMesh->volume();
    int j = 0;
    while (abs(res)>tol){
        _gradient->reproject(res);
        res = _startingVol -_DMesh->volume();

        j++;
        if (j>_maxConstraintSteps){
            printf("Warning: Too many steps in constraint satisfaction\n");
            break;
        }
    }
    return res;

}


double ShapeOptimizer::gradientDesent(int n){ // do n gradient desent steps
    double area = _DMesh->area();
    double res = 0;
    double lastArea = 0;
    double dArea = 0;
    fprintf(stdout,"At Start: volume = %f \t area = %f \n",_DMesh->volume(),_DMesh->area());
    for (int i = 0; i<n; i++){
        int j = 0;
        gradientDesentStep();
        lastArea = area;
        area = _DMesh->area();
        dArea = area - lastArea;
        


        fprintf(stdout,"Step %d: volume = %f \t area = %f delta area = %f\n",i+1,_DMesh->volume(),area,dArea);
        if (abs(dArea)<_dAtol){break;}
    
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

