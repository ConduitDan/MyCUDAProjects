#include "ShapeOptimizer.hpp"

int main(){
    ShapeOptimizer myOptimizer("cube.mesh");
    myOptimizer.gradientDesent(1000);

    myOptimizer.printMesh("hopefullyASphere.mesh");

    return 1;



}