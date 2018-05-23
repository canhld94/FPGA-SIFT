#include "_sift_kernel.h"
/********************************************************************************************************************/
/*              dogOct kernel and sub modules                                                                       */
/*              INPUT: pipes, size of image                                                                         */
/*              OUTPUT: poiter to output DoG                                                                        */
/********************************************************************************************************************/

void CopyOutputImageToDRAM( global data_t *output,global data_t *_output_dog,int width,int height);
__attribute__ ((xcl_reqd_work_group_size(1,1,1)))
__attribute__ ((xcl_dataflow))
__kernel void dog_oct(   int height,
                        int width,
                        global data_t *output 
                    )
{

    __attribute__ ((xcl_pipeline_loop))
    DOG_LOOP_0: for(int i = 0; i < width*height; ++i){
    	__local conv_data0,conv_data1,dog_data0;
// Read from correcsponding pipes
    	read_pipe_block(pipe_scale_0,&conv_data0);
    	read_pipe_block(pipe_scale_1,&conv_data1);
// Subtraction
    	dog_data0 = conv_data0 - conv_data1;
// Write to correcsponding bram 
    	_output_dog_0[i] = dog_data0;
    }
    __attribute__ ((xcl_pipeline_loop))
    DOG_LOOP_1: for(int i = 0; i < width*height; ++i){
    	__local conv_data1,conv_data2,dog_data1;
    	// Read from correcsponding pipes
    	    	read_pipe_block(pipe_scale_1_1,&conv_data1);
    	    	read_pipe_block(pipe_scale_2,&conv_data2);
    	// Subtraction
    	    	dog_data1 = conv_data1 - conv_data2;
    	// Write to correcsponding bram
    	    	_output_dog_1[i] = dog_data1;
    }
    __attribute__ ((xcl_pipeline_loop))
    DOG_LOOP_2: for(int i = 0; i < width*height; ++i){
    	__local conv_data2,conv_data3,dog_data2;
    	// Read from correcsponding pipes
    	    	read_pipe_block(pipe_scale_2_1,&conv_data2);
    	    	read_pipe_block(pipe_scale_3,&conv_data3);
    	// Subtraction
    	    	dog_data2 = conv_data2 - conv_data3;
    	// Write to correcsponding bram
    	    	_output_dog_2[i] = dog_data2;
    }
    __attribute__ ((xcl_pipeline_loop))
    DOG_LOOP_3: for(int i = 0; i < width*height; ++i){
    	__local conv_data3,conv_data4,dog_data3;
    	// Read from correcsponding pipes
    	    	read_pipe_block(pipe_scale_3_1,&conv_data3);
    	    	read_pipe_block(pipe_scale_4,&conv_data4);
    	// Subtraction
    	    	dog_data3 = conv_data3 - conv_data4;
    	// Write to correcsponding bram
    	    	_output_dog_3[i] = dog_data3;
    }
// copy data to DRAM
    CopyOutputImageToDRAM(output, _output_dog_0, width, height);
    CopyOutputImageToDRAM(output + width*height, _output_dog_1, width, height);
    CopyOutputImageToDRAM(output + 2*width*height, _output_dog_2, width, height);
    CopyOutputImageToDRAM(output + 3*width*height, _output_dog_3, width, height);
}

/********************************************************************************************************************/
/*              Sub modules prototype here                                                                          */
/********************************************************************************************************************/

void CopyOutputImageToDRAM( global data_t *output,
                             global data_t *_output_dog,
                             int width, 
                             int height)
{
    __attribute__ ((xcl_pipeline_loop))
    COPY_OUTPUT_LOOP: for(int i = 0; i < width*height; ++i){
        output[i] = _output_dog[i];
    }
}
