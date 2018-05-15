#include "_sift_kernel.h"

__kernel __attribute__ ((xcl_reqd_work_group_size(1,1,1))) void conv_oct(/*__local float* Shift_reg*/){ // input img and filter values are on-chip global; you can access these value directly instead of getting these values by arguments

	printf("Executing conv_oct kernel...\n");	

	//conv_scale_0();
	//conv_scale_1();
	//conv_scale_2();
	//...

}

void conv_scale_0(){

	float output_data = 0.0;

	// Doing Convolution 

	// Write results in pipe
	write_pipe(pipe_scale_0, &output_data);	

}
