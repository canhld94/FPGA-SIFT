#include "_sift_kernel.h"
/********************************************************************************************************************/
/*              dogOct kernel and sub modules                                                                       */
/*              INPUT: pipes, size of image                                                                         */
/*              OUTPUT: poiter to output DoG                                                                        */
/********************************************************************************************************************/

__attribute__ ((xcl_reqd_work_group_size(1,1,1)))
__attribute__ ((xcl_dataflow))
__kernel void dog_oct(   int height,
                        int width,
                        global data_t *output 
                    )
{

    __attribute__ ((xcl_pipeline_loop))
    DOG_LOOP_0: for(int i = 0; i < width*height; ++i){
// Read from correcsponding pipes
// Subtraction
// Write to correcsponding bram 
    }
    __attribute__ ((xcl_pipeline_loop))
    DOG_LOOP_1: for(int i = 0; i < width*height; ++i){

    }
    __attribute__ ((xcl_pipeline_loop))
    DOG_LOOP_2: for(int i = 0; i < width*height; ++i){

    }
    __attribute__ ((xcl_pipeline_loop))
    DOG_LOOP_3: for(int i = 0; i < width*height; ++i){

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
