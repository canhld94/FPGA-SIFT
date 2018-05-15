/* Written by eyoh
 *
 * 1) InputToOnchip
 * - move input image to on chip global memory
 * - you can modify the code to apply async_work_group_copy()
 * - I simply implement the kernel in a way that one work-item move input_img pixel by pixel in sequential
 *
 * 2) FilterToOnchip
 * - move gaussian filters to on chip global memory
 * - you can modify the code to apply async_work_group_copy()
 * - I simply implement the kernel in a way that one work-item move input_img pixel by pixel in sequential
 *
 *
 * NOTICE >> 1),2) is for seperating the data movement and processing (as mentioned by Canh)
 *
 *
 * 3) Shfit_register
 * - you should define SHIFT_REG_WIDTH in a proper size
 * - input img was moved to on-chip global, so you can access these value directly (don't need to get these value as argument)
 * - You have to consider zero padding for convolution. 
 * - You should care about the downsizing of img according to octave
 * - Shfit_reg_begin() : at first, the shift register is empty, so you need to push values to the shift register untill values, which are needed for the first convolution, are pushed on it.
 * - Shift_reg_loop() : if the shift register is not empty, you can push one pixel per each cycle
 * - Shift_reg_end() : is there a difference with Shift_reg_loop ? i don know.. you have to think about it.
 * 
 * - But.. I'm not sure wether the shift register is needed or not.
 * - The objective of shift register is caching input image to be used.
 * - However, if we move whole input image to on-chip, it means we already cache all the input image on-chip.
 * - we can access each pixel directly (not using shift register)...
 * 
 * 
 *
 * 4) conv_oct
 * - conv_oct includes convolutions for each scales
 * - you should implement convolution of each scale independently, (e.g. conv_scale_#)
 * - If you write one function for convolution and reuse it for all scale, the compiler will synthesis only one hardware for it and each convolution kernel are exectued in sequential
 * - input img and filter values are on-chip global, so you can access these value directly (don't need to get these value as arguments)
 *
 * 5) dog_oct
 * - dog_oct includes matrix subtraction for each scales
 * - you should implement dog of each scale independently, (e.g. dog_scale_#)
 * - If you write one function for dog and reuse it for all scale, the compiler will synthesis only one hardware for it and each dog kernel are exectued in sequential
 * - input img and filter values are on-chip global, so you can access these value directly (don't need to get these value as arguments)
 * - the output value of dog will write down on on-chip global buffer in temporary 
 *
 * 6) OutputFromOnchip
 * - move temporary dog output value (on-chip global) to DDR4 (off-chip global) for accessing by the host
 *
 * NOTICE >> It is not possible that one kernel write on and read from the same kernel.
 * 	     One pipe should be shared by two kernel, read-kernel and write-kernel.
 *	     So, we cannot assemble conv_oct and dog_oct in a one kernel, like __kernel Build_Dog_pyramid.
 * 	     As a result, I will write down a host code to call kernels like in the below way
 * 		for (each octave){
 *			InputToOnchip();
 *			FilterToOnchip();			
 *
 *			Shift_reg_loop();
 * 			
 * 			conv_oct();
 *			dog_oct();
 *			OutputFromOnchip();
 *		}
 * 	    
 * 	    If we use shift register, I think we have to insert a pixel-loop inside of the octave-loop
 *	    because the shift register update one pixel value per cycle,
 *	    other kernels (conv, dog) have to calculate one pixel output value per cycle.
 * 	    To do so, we have to loop the one-octave procedure for total-pixel times....
 * 	    I'm not sure wether it is an efficient way or not. :(
 */

#ifndef __SIFT_KERNEL_H_
#define __SIFT_KERNEL_H_

//// MACRO DEFINITION ////
#define MAX_FLEN 25
#define WIDTH 320
#define HEIGHT 240
#define SCALES 6
#define SHIFT_REG_WIDTH WIDHT // you can modify it 
#define FILTER_LEN_SCALE_0 9
#define FILTER_LEN_SCALE_1 11
#define FILTER_LEN_SCALE_2 13
#define FILTER_LEN_SCALE_3 15
#define FILTER_LEN_SCALE_4 21


//// ON-CHIP GLOBAL VARIABLE DECLARATION ////

// on-chip global memory for input image 
global float _input_img[WIDTH*HEIGHT];

// on-chip global for gaussian filter
// We can modify it by hardcoding the filter values in each kernel
global float _gaussianFilter_scale_0[FILTER_LEN_SCALE_0*FILTER_LEN_SCALE_0];
global float _gaussianFilter_scale_1[FILTER_LEN_SCALE_1*FILTER_LEN_SCALE_1];
global float _gaussianFilter_scale_2[FILTER_LEN_SCALE_2*FILTER_LEN_SCALE_2];
global float _gaussianFilter_scale_3[FILTER_LEN_SCALE_3*FILTER_LEN_SCALE_3];
global float _gaussianFilter_scale_4[FILTER_LEN_SCALE_4*FILTER_LEN_SCALE_4];

// on-chip global memory for output DoG
global float _output_dog[WIDTH*HEIGHT];

//// PIPE DECLARATION ////

// for storing output of convolution module

pipe float pipe_scale_0 __attribute__((xcl_reqd_pipe_depth(512)));
pipe float pipe_scale_1 __attribute__((xcl_reqd_pipe_depth(256)));
pipe float pipe_scale_2 __attribute__((xcl_reqd_pipe_depth(128)));
pipe float pipe_scale_3 __attribute__((xcl_reqd_pipe_depth(64)));
pipe float pipe_scale_4 __attribute__((xcl_reqd_pipe_depth(32)));

//// FUNCTION DECLARATION ////
__kernel __attribute__ ((xcl_req_work_group_size(1,1,1))) void InputToOnchip(global float* input);

__kernel __attribute__ ((xcl_req_work_group_size(1,1,1))) void FilterToOnchip_scale_0(global float* filter); // We can implement these functions for all scales in one kernel 
__kernel __attribute__ ((xcl_req_work_group_size(1,1,1))) void FilterToOnchip_scale_1(global float* filter);
__kernel __attribute__ ((xcl_req_work_group_size(1,1,1))) void FilterToOnchip_scale_2(global float* filter);
__kernel __attribute__ ((xcl_req_work_group_size(1,1,1))) void FilterToOnchip_scale_3(global float* filter);
__kernel __attribute__ ((xcl_req_work_group_size(1,1,1))) void FilterToOnchip_scale_4(global float* filter);

__kernel __attribute__ ((xcl_req_work_group_size(1,1,1))) void Shift_reg_begin();
__kernel __attribute__ ((xcl_req_work_group_size(1,1,1))) void Shift_reg_loop();
__kernel __attribute__ ((xcl_req_work_group_size(1,1,1))) void Shift_reg_end();

//__kernel __attribute__ ((xcl_req_work_group_size(1,1,1))) void Build_DoG_pyramid(__global float* _output);
__kernel __attribute__ ((xcl_req_work_group_size(1,1,1))) void conv_oct(/*__local float* Shift_reg*/);
__kernel __attribute__ ((xcl_req_work_group_size(1,1,1))) void dog_oct();

#endif