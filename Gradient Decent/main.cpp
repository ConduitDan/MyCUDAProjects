#include "ShapeOptimizer.hpp"

/*
Next steps for openCL:
Split device mesh into 2 sub classes (DeviceMeshCUDA DeviceMeshCL)
Save for gradient
make a factory
have optimzer call the facotry which switches what kind of DMesh and Grandients it hands out. 
So we have
[ ] Figue out make file
[ ] Figure out #ifdef USE_BUILD_X
[ ] Split mesh
[ ] split Grandiet
[ ] rewrite kernals
[ ] make Factory
[ ] hook up facotry to Optimzier*/


int main(int nargs, char** argv){
    // takes in a filename and a number of steps to take
    
    ShapeOptimizer myOptimizer(argv[1]);
    myOptimizer.gradientDesent(atoi(argv[2]));

    myOptimizer.printMesh("hopefullyASphere.mesh");

    return 1;



}