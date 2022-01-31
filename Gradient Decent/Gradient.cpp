#include "Gradient.hpp"

Gradient::Gradient(DeviceMesh *inMesh, DeviceAPI* GPUAPIin){

	_GPU = GPUAPIin;
    
	_myMesh = inMesh;
    unsigned int numVert = _myMesh->get_numVert();
    unsigned int numFacet = _myMesh->get_numFacets();


	_gradAFacet = UniqueDevicePtr<double>(_GPU);
	_gradVFacet = UniqueDevicePtr<double>(_GPU);

	_gradAVert = UniqueDevicePtr<double>(_GPU);
	_gradVVert = UniqueDevicePtr<double>(_GPU);


	_force = UniqueDevicePtr<double>(_GPU);
	_scratch = UniqueDevicePtr<double>(_GPU);


    _gradAFacet.allocate(numFacet * 3 * 3);
    _gradVFacet.allocate(numFacet * 3 * 3);
    
	_gradAVert.allocate(numVert * 3 * 3);
    _gradVVert.allocate(numVert * 3 * 3);
        
	_force.allocate(numVert * 3 * 3);
    // the scratc vector is used as scrach for taking the dot products for projection,
    // so it needs to be padded to a multiple of twice the block size so we can effiecntly sum it
    unsigned int bufferedSize = ceil(numVert * 3 /(2.0*_GPU->get_blockSize()))*2*_GPU->get_blockSize();
    _scratch.allocate(bufferedSize);
}

void Gradient::calc_force(){
    calc_gradA();
	calc_gradV();
	project_to_force();
}

void Gradient::calc_gradA(){
    // first calculate the gradient on the facets

	// double* gradAVert2 = new double[_myMesh->get_numVert()*3];
	// double* gradAVert = new double[_myMesh->get_numVert()*3];


	// _GPU->copy_to_host(gradAVert,_gradAVert.get(),sizeof(double)*3*_myMesh->get_numVert());


	switch (_noF2V){
		case 1:
			_GPU->area_gradient(&_gradAFacet,_myMesh->get_facets(),_myMesh->get_vert(),_myMesh->get_numFacets());
			_GPU->facet_to_vertex(&_gradAVert,&_gradAFacet,_myMesh->get_vertToFacet(),_myMesh->get_vertIndexStart(),_myMesh->get_numVert());
			break;
		case 2:
			_GPU->area_gradient2(&_gradAVert,_myMesh->get_facets(),_myMesh->get_vert(),\
							_myMesh->get_numFacets(),_myMesh->get_numVert());
			break;
		case 3:
			_GPU->area_gradient3(&_gradAVert,_myMesh->get_facets(),_myMesh->get_vert(),\
							_myMesh->get_numFacets(),_myMesh->get_numVert());
	}





	// double ele1;
	// double ele2;
	// double ele3;
	// int VERTTOCHECK = 1;
	// _GPU->copy_to_host(&ele1,_gradAVert.get()+3*VERTTOCHECK,sizeof(double));
	// _GPU->copy_to_host(&ele2,_gradAVert.get()+1+3*VERTTOCHECK,sizeof(double));
	// _GPU->copy_to_host(&ele3,_gradAVert.get()+2+3*VERTTOCHECK,sizeof(double));

	// printf("for vertex %d i got {%f,%f,%f}\n",VERTTOCHECK,ele1,ele2,ele3);
	// printf("I should have goten the sum of:\n");

	// // copy the vert index start [1]
	// unsigned int finishIndex;
	// unsigned int facetNo;
	// unsigned int startIndex;
	// _GPU->copy_to_host(&startIndex,_myMesh->get_vertIndexStart()->get()+VERTTOCHECK,sizeof(unsigned int));

	// _GPU->copy_to_host(&finishIndex,_myMesh->get_vertIndexStart()->get()+1+VERTTOCHECK,sizeof(unsigned int));
	// for (int i=startIndex; i<finishIndex; i++){
	// 	//grab facet to vertex i
	// 	_GPU->copy_to_host(&facetNo,_myMesh->get_vertToFacet()->get()+i,sizeof(unsigned int));
	// 	//facetNo now had a facet that has vertex 0
	// 	// we now need to copy the 3 elemets in _gradAFacet for it
	// 	_GPU->copy_to_host(&ele1,_gradAFacet.get()+3*facetNo,sizeof(double));
	// 	_GPU->copy_to_host(&ele2,_gradAFacet.get()+3*facetNo+1,sizeof(double));
	// 	_GPU->copy_to_host(&ele3,_gradAFacet.get()+3*facetNo+2,sizeof(double));
	// 	printf("{%f,%f,%f}\n",ele1,ele2,ele3);


	// }
	// _GPU->copy_to_host(gradAVert2,_gradAVert.get(),sizeof(double)*3*_myMesh->get_numVert());



	// for (int i = 0; i < _myMesh->get_numVert()*3; i++){
	// 	if(fabs(gradAVert2[i]-gradAVert[i])>1e-10){
	// 		printf("found a descrepency of size %e\n",gradAVert2[i]-gradAVert[i]);
	// 		break;
	// 	}
	// }


	//}
	//make sure they're equal


}

void Gradient::calc_gradV(){
    // first calculate the gradient on the facets


	switch (_noF2V){
		case 1:
		    _GPU->volume_gradient(&_gradVFacet,_myMesh->get_facets(),_myMesh->get_vert(),_myMesh->get_numFacets());
   			_GPU->facet_to_vertex(&_gradVVert,&_gradVFacet,_myMesh->get_vertToFacet(),_myMesh->get_vertIndexStart(),_myMesh->get_numVert());
			break;
		case 2:
			_GPU->volume_gradient2(&_gradVVert,_myMesh->get_facets(),_myMesh->get_vert(),\
					_myMesh->get_numFacets(),_myMesh->get_numVert());
			break;
		case 3:
			_GPU->volume_gradient3(&_gradVVert,_myMesh->get_facets(),_myMesh->get_vert(),\
					_myMesh->get_numFacets(),_myMesh->get_numVert());


	}

		
	// double* gradVVert2 = new double[_myMesh->get_numVert()*3];
	// double* gradVVert = new double[_myMesh->get_numVert()*3];
	// _GPU->copy_to_host(gradVVert,_gradVVert.get(),sizeof(double)*3*_myMesh->get_numVert());
	// _GPU->copy_to_host(gradVVert2,_gradVVert.get(),sizeof(double)*3*_myMesh->get_numVert());
	// for (int i = 0; i < _myMesh->get_numVert()*3; i++){
	// 	if(fabs(gradVVert2[i]-gradVVert[i])>1e-10){
	// 		printf("found a descrepency of size %e\n",gradVVert2[i]-gradVVert[i]);
	// 		break;
	// 	}
	// }
	//make sure they're equal

}
void Gradient::project_to_force(){
    double numerator = _GPU->dotProduct(&_gradAVert,&_gradVVert,&_scratch,_myMesh->get_numVert() * 3);
    double denominator = _GPU->dotProduct(&_gradVVert,&_gradVVert,&_scratch,_myMesh->get_numVert() * 3 );
    _GPU->project_force(&_force,&_gradAVert,&_gradVVert,numerator/abs(denominator),_myMesh->get_numVert() * 3);
    
}

void Gradient::reproject(double res){
    calc_gradV();
    // do the force inner product

    double m = _GPU->dotProduct(&_gradVVert,&_gradVVert,&_scratch,_myMesh->get_numVert()*3);
    double sol = res/m;
    //move and scale (scale = sol, dir = gradV)
    _GPU->add_with_mult(_myMesh->get_vert(),&_gradVVert,sol,_myMesh->get_numVert()*3);
    

}
