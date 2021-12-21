#include "Gradient.hpp"

Gradient::Gradient(DeviceMesh *inMesh){
    _myMesh = inMesh;
    unsigned int numVert = _myMesh->get_numVert();
    unsigned int numFacet = _myMesh->get_numFacets();


    _gradAFacet.reset((double*)GPU->allocate(numFacet * 3 * 3 * sizeof(double)));
    _gradAVert.reset((double*)GPU->allocate(numVert * 3 * 3 * sizeof(double)));
    _gradVFacet.reset((double*)GPU->allocate(numFacet * 3 * 3 * sizeof(double)));
    _gradVVert.reset((double*)GPU->allocate(numVert * 3 * 3 * sizeof(double)));
    _force.reset((double*)GPU->allocate(numVert * 3 * 3 * sizeof(double)));
    // the scratc vector is used as scrach for taking the dot products for projection,
    // so it needs to be padded to a multiple of twice the block size so we can effiecntly sum it
    unsigned int bufferedSize = ceil(numVert * 3 /(2.0*_myMesh->get_blockSize()))*2*_myMesh->get_blockSize();
    _scratch.reset((double*)GPU->allocate(bufferedSize * sizeof(double)));
}

void Gradient::calc_force(){
    calc_gradA();
	calc_gradV();
	project_to_force();
}

void Gradient::calc_gradA(){
    // first calculate the gradient on the facets
    GPU->area_gradient(_gradAFacet.get(),_myMesh->get_facets(),_myMesh->get_vert(),_myMesh->get_numFacets());
    GPU->facet_to_vertex(_gradAFacet.get(),_gradAVert.get(),_myMesh->get_vertToFacet(),_myMesh->get_vertIndexStart(),_myMesh->get_numVert());

}

void Gradient::calc_gradV(){
    // first calculate the gradient on the facets
    GPU->volume_gradient(_gradVFacet.get(),_myMesh->get_facets(),_myMesh->get_vert(),_myMesh->get_numFacets());
    GPU->facet_to_vertex(_gradVFacet.get(),_gradVVert.get(),_myMesh->get_vertToFacet(),_myMesh->get_vertIndexStart(),_myMesh->get_numVert());

}
void Gradient::project_to_force(){
    double numerator = GPU->dotProduct(_gradAVert.get(),_gradVVert.get(),_scratch.get(),_myMesh->get_numVert() * 3,_myMesh->get_blockSize() );
    double denominator = GPU->dotProduct(_gradVVert.get(),_gradVVert.get(),_scratch.get(),_myMesh->get_numVert() * 3,_myMesh->get_blockSize() );

    unsigned int numberOfBlocks = ceil(_myMesh->get_numVert() * 3 / (float) _myMesh->get_blockSize());
    GPU->project_force(_force.get(),_gradAVert.get(),_gradVVert.get(),numerator/abs(denominator),_myMesh->get_numVert() * 3);
    
}

void Gradient::reproject(double res){
    calc_gradV();
    // do the force inner product

    double m = GPU->dotProduct(_gradVVert.get(),_gradVVert.get(),_scratch.get(),_myMesh->get_numVert()*3,_myMesh->get_blockSize());
    double sol = res/m;
    //move and scale (scale = sol, dir = gradV)
    unsigned int numberOfBlocks = ceil(_myMesh->get_numVert() * 3 / (float) _myMesh->get_blockSize());
    GPU->add_with_mult(_myMesh->get_vert(),_gradVVert.get(),sol,_myMesh->get_numVert()*3);
    

}
