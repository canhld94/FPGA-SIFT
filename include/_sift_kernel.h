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
#define SCALES 5
#define SHIFT_REG_WIDTH WIDHT // you can modify it 
#define FLEN0 9
#define FLEN1 11
#define FLEN2 13
#define FLEN3 15
#define FLEN4 21

typedef float data_t ;

//// ON-CHIP GLOBAL VARIABLE DECLARATION ////

// on-chip global memory for input image 
global data_t _input_img[WIDTH*HEIGHT];
global data_t _intm_img[WIDTH*HEIGHT]; // typo correction WITH -> WIDTH


// on-chip global memory for output DoG
// We need 4 buffer to store 4 dog image. We need to discuss more about this because we cannot do 
// this when image size increase. I'm having plan to use pipe as a buffer between computing unit and DRAM.
// The idea is result will be written to pipe, and DMA module will read from that pipe and write to output
// pointer.

// how about defining _output_dog size as WIDHT (or other unit of size)
// and whenever one line of dog image is calculated, move the output line of dog to DRAM
// In this case, we need to consider synchronization problem.
// we don't know each dog scales are finished calculation of one line of output at certain time 
// oh, yes... due to this problem, it will be better to use pipe for synchronization. ㅋㅋ
// you are right canh
// - eyoh
global data_t _output_dog_0[WIDTH*HEIGHT];
global data_t _output_dog_1[WIDTH*HEIGHT];
global data_t _output_dog_2[WIDTH*HEIGHT];
global data_t _output_dog_3[WIDTH*HEIGHT];


// PIPE for storing output of convolution module

//Since a loop of Dog kernel needs outputs from two different convolution module,
//convolution module for scale 1,2,3 must send data twice via pipe_scale_*_1
//therefore i added three more pipes
//-kclee
pipe data_t pipe_scale_0 __attribute__((xcl_reqd_pipe_depth(128)));
pipe data_t pipe_scale_1 __attribute__((xcl_reqd_pipe_depth(128)));
pipe data_t pipe_scale_1_1 __attribute__((xcl_reqd_pipe_depth(128)));
pipe data_t pipe_scale_2 __attribute__((xcl_reqd_pipe_depth(128)));
pipe data_t pipe_scale_2_1 __attribute__((xcl_reqd_pipe_depth(128)));
pipe data_t pipe_scale_3 __attribute__((xcl_reqd_pipe_depth(128)));
pipe data_t pipe_scale_3_1 __attribute__((xcl_reqd_pipe_depth(128)));
pipe data_t pipe_scale_4 __attribute__((xcl_reqd_pipe_depth(128)));

//// FUNCTION DECLARATION ////
// __kernel __attribute__ ((xcl_req_work_group_size(1,1,1))) void InputToOnchip(global data_t* input);

// __kernel __attribute__ ((xcl_req_work_group_size(1,1,1))) void FilterToOnchip_scale_0(global data_t* filter); // We can implement these functions for all scales in one kernel 
// __kernel __attribute__ ((xcl_req_work_group_size(1,1,1))) void FilterToOnchip_scale_1(global data_t* filter);
// __kernel __attribute__ ((xcl_req_work_group_size(1,1,1))) void FilterToOnchip_scale_2(global data_t* filter);
// __kernel __attribute__ ((xcl_req_work_group_size(1,1,1))) void FilterToOnchip_scale_3(global data_t* filter);
// __kernel __attribute__ ((xcl_req_work_group_size(1,1,1))) void FilterToOnchip_scale_4(global data_t* filter);

// __kernel __attribute__ ((xcl_req_work_group_size(1,1,1))) void Shift_reg_begin();
// __kernel __attribute__ ((xcl_req_work_group_size(1,1,1))) void Shift_reg_loop();
// __kernel __attribute__ ((xcl_req_work_group_size(1,1,1))) void Shift_reg_end();

//__kernel __attribute__ ((xcl_req_work_group_size(1,1,1))) void Build_DoG_pyramid(__global data_t* _output);
__kernel __attribute__ ((xcl_req_work_group_size(1,1,1))) void conv_oct(global data_t *input,int height,int width,bool from_bram);
__kernel __attribute__ ((xcl_req_work_group_size(1,1,1))) void dog_oct(int height,int width,global data_t *output ); // Argument setting

#endif
