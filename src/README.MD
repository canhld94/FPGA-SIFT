
There are some different with previous skeleton:
1. Now we hardcode the filter coeficient in FPGA, then don't need to transfer filter value from dram to filter buffer. If you are implementing conv2d moudle, plesase notice!!

2. All memory objects: input image buffer, output image buffer, pipe are declarated in _sift_kernel.h, please take a look at this file, it's helpful

3. Now we only work with 2 kernel: conv_oct and dog_oct, mean 2 files _conv_oct.cl and _dog_oct.cl, please don't modify other files

4. Refer to these instructions before modify the code or commit:

4.1. In conv_oct kernel: Now need to implement five conv2d modules, refer to conv2d_s0, conv2d_s1, conv2d_s2, conv2d_s3, conv2d_s4

4.2. In dogOct kernel: You need to complete all the for loops to calculate DoG image
  *	 There are 4 dependent loops (DOG_LOOP_0, DOG_LOOP_1, DOG_LOOP_2, DOG_LOOP_3).
  *	 Each loops are corresponding to dog_scale_0, dog_scale_1, ...

4.3. Whoever working with conv2d module pleas notice: output of conv2d_s2 need to be store for next octave, refer to instruction in kernel skeleton 

4.4. Please, work with your modules only, DO NOT modify other codes and DO NOT change the module interface

4.5. If you can, please label all loop in your code with a understandable label, it's helpful for debug and simulation

4.6. If you have any problem, please create issue on Github or ask Eun Young :)))))))))))))))))))))))))))

Execution model in host code:
*	for(int = 0; i < nOctaves; ++i)
*      			dowsizeimage();
*			conv_oct();
*			dog_oct(); 
*		}  	
Hope it works!!   
