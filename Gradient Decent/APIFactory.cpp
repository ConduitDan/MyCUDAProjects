#include "APIFactory.hpp"


DeviceAPI* APIFactory::myAPI = nullptr;

DeviceAPI* APIFactory::get_API(int blockSize){
	if (myAPI){
		if (myAPI->get_blockSize() != blockSize) myAPI->set_blockSize(blockSize);
	}
	else {
		myAPI = new API(blockSize);
	}
	return myAPI;

}

#undef API


// set the pointer to null by default
