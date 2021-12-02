#include "ShapeOptimizer.hpp"

int main(int nargs, char** argv){
    // takes in a filename and a number of steps to take
    
    ShapeOptimizer myOptimizer(argv[1]);
    myOptimizer.gradientDesent(atoi(argv[2]));

    myOptimizer.printMesh("hopefullyASphere.mesh");

    return 1;



}