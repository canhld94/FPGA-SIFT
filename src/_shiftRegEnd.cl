#include "_sift_kernel.h"

__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void Shift_reg_end(){ // input img are moved to on-chip global so you can access directly

	printf("Executing Shift_reg_end kernel ...\n");

}