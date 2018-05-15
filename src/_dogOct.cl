#include "_sift_kernel.h"

__kernel __attribute__ ((xcl_reqd_work_group_size(1,1,1))) void dog_oct(/*__local float* Shift_reg*/){ // input img and filter values are on-chip global; you can access these value directly instead of getting these values by arguments

	printf("Executing dog_oct kernel...\n");

	//dog_scale_0();
	//dog_scale_1();
	//dog_scale_2();
	//...

}

void dog_scale0(){
	
	float output_data = 0.0;
	// Read input from pipe
	float input_scale_0 = 0.0;
	float input_scale_1 = 0.0;
	read_pipe(pipe_scale_0, &input_scale_0);
	read_pipe(pipe_scale_1, &input_scale_1);

	// Doing DoG

	// Write results
	int idx = 0;
	_output_dog[idx] = output_data; // you should calculate proper idx
}
