#include "_sift_kernel.h"

__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void FilterToOnchip_scale_1(
	__global float* filter)
{
	__attribute__((opencl_unroll_hint))
	for(int i=0; i<FILTER_LEN_SCALE_1*FILTER_LEN_SCALE_1; i++){
		_gaussianFilter_scale_1[i] = filter[i];
	}

	// Check 
	printf("=======gaussianFilter_scale_1=========\n");
	for(int i=0; i<FILTER_LEN_SCALE_1; i++){
		printf("filter_1 row %d:", i);
		for(int j=0; j<FILTER_LEN_SCALE_1;j++){
		printf("%f ", _gaussianFilter_scale_1[i*FILTER_LEN_SCALE_1+j]);
		}
		printf("\n");
	}
}
__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void FilterToOnchip_scale_2(
	__global float* filter)
{
	__attribute__((opencl_unroll_hint))
	for(int i=0; i<FILTER_LEN_SCALE_2*FILTER_LEN_SCALE_2; i++){
		_gaussianFilter_scale_2[i] = filter[i];
	}

	// Check 
	printf("=======gaussianFilter_scale_2=========\n");
	for(int i=0; i<FILTER_LEN_SCALE_2; i++){
		printf("filter_2 row %d:", i);
		for(int j=0; j<FILTER_LEN_SCALE_2;j++){
		printf("%f ", _gaussianFilter_scale_2[i*FILTER_LEN_SCALE_2+j]);
		}
		printf("\n");
	}


}


__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void FilterToOnchip_scale_3(
	__global float* filter)
{
	__attribute__((opencl_unroll_hint))
	for(int i=0; i<FILTER_LEN_SCALE_3*FILTER_LEN_SCALE_3; i++){
		_gaussianFilter_scale_3[i] = filter[i];
	}

	// Check 
	printf("=======gaussianFilter_scale_3=========\n");
	for(int i=0; i<FILTER_LEN_SCALE_3; i++){
		printf("filter_3 row %d:", i);
		for(int j=0; j<FILTER_LEN_SCALE_3;j++){
		printf("%f ", _gaussianFilter_scale_3[i*FILTER_LEN_SCALE_3+j]);
		}
		printf("\n");
	}

}

__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void FilterToOnchip_scale_4(
	__global float* filter)
{
	__attribute__((opencl_unroll_hint))
	for(int i=0; i<FILTER_LEN_SCALE_4*FILTER_LEN_SCALE_4; i++){
		_gaussianFilter_scale_4[i] = filter[i];
	}

	// Check 
	printf("=======gaussianFilter_scale_4=========\n");
	for(int i=0; i<FILTER_LEN_SCALE_4; i++){
		printf("filter_4 row %d:", i);
		for(int j=0; j<FILTER_LEN_SCALE_4;j++){
		printf("%f ", _gaussianFilter_scale_4[i*FILTER_LEN_SCALE_4+j]);
		}
		printf("\n");
	}

}