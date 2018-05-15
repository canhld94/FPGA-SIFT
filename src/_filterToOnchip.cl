#include "_sift_kernel.h"

__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void FilterToOnchip_scale_0(
	__global float* filter)
{
	__attribute__((opencl_unroll_hint))
	for(int i=0; i<FILTER_LEN_SCALE_0*FILTER_LEN_SCALE_0; i++){
		_gaussianFilter_scale_0[i] = filter[i];
	}

	// Check 
	printf("=======gaussianFilter_scale_0=========\n");
	for(int i=0; i<FILTER_LEN_SCALE_0; i++){
		printf("filter_0 row %d:", i);
		for(int j=0; j<FILTER_LEN_SCALE_0;j++){
		printf("(%d) %f ",i*FILTER_LEN_SCALE_0+j, _gaussianFilter_scale_0[i*FILTER_LEN_SCALE_0+j]);
		}
		printf("\n");
	}
}
