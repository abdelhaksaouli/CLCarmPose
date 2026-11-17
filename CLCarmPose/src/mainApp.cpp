#include "gpuppfstimator.h"

#define GPU_MAX_HEAP_SIZE 100

int main()
{

	GPUPPFEstimator estimator;
	estimator.init_device();
	estimator.trainModel();
	estimator.setUpScene();
	estimator.PPFGPURegistration();
	
	return 0;
}