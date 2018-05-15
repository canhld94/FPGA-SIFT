#include "_sift_kernel.h"

__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void InputToOnchip(__global float* input)
{
	for(int i=0; i<WIDTH*HEIGHT; i++){
		_input_img[i] = input[i];	
	}	
}