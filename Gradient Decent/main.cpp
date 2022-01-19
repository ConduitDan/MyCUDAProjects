#include "ShapeOptimizer.hpp"

/*
Next steps for openCL:
Split device mesh into 2 sub classes (DeviceMeshCUDA DeviceMeshCL)
Save for gradient
make a factory
have optimzer call the facotry which switches what kind of DMesh and Grandients it hands out. 
So we have
[x] Figue out make file
[x] Figure out #ifdef USE_BUILD_X
[x] Split mesh
[x] split Grandiet
[ ] rewrite kernals
[x] make Factory
[x] hook up facotry to Optimzier*/


int main(int nargs, char** argv){
    // takes in a filename and a number of steps to take
    try{
    ShapeOptimizer myOptimizer(argv[1]);
    myOptimizer.gradientDesent(atoi(argv[2]));

    myOptimizer.printMesh("Sphere.mesh");

    return 0;
    }
    catch(const char* e){
        printf("failed with error %s\n",e);
        return -1;
    }


}