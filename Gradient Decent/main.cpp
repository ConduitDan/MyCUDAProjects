#include "ShapeOptimizer.hpp"

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