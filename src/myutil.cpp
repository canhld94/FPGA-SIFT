#include "myutil.h"
#include "oclHelper.h"
#include <string.h>
#include <CL/cl.h>
#include <stdio.h>

void checkErrorStatus(cl_int error, const char* message)
{
  if (error != CL_SUCCESS)
  {
    printf("%s\n", message) ;
    printf("%s\n", oclErrorCode(error)) ;
    exit(0) ;
  }
}

void rgb2float(cl_uchar* srcImg, cl_float *(&dstImg), int width, int height){

	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
			dstImg[j+width*i] = 0.114f*srcImg[3*i + 3*width*i] + 0.587f*srcImg[3*j+1+3*width*i] + 0.299f*srcImg[3*j+2+3*width*i];
		}
	}

}

void clCreateKernel_withErrorCheck(cl_kernel& dstKernelVar, oclSoftware& software, char* kernel){
	cl_int err = 0;
	char msg[200];
	dstKernelVar = clCreateKernel(software.mProgram,kernel,&err);
	sprintf(msg, "Unable to create kernel: %s", kernel);
	checkErrorStatus(err, msg);

}

void setArguments(cl_kernel mKernel, char *input_type, ...){
        va_list ap;
        cl_int err = 0;
        int num = 0;

        va_start(ap, input_type);
        char *ptr = strtok(input_type, " ");

        while (ptr != NULL)
        {
                if(!strcmp(ptr, "%f")) {
                        float *temp = (float *)va_arg(ap, double*);
                        err = clSetKernelArg(mKernel, num, sizeof(float), temp);
                        checkErrorStatus(err, "Unable to set argument");
                }
                else if(!strcmp(ptr, "%d")) {
                        int *temp = (int *)va_arg(ap, int*);
                        err = clSetKernelArg(mKernel, num, sizeof(int), temp);
                        checkErrorStatus(err, "Unable to set argument");
                }
                else if(!strcmp(ptr, "%cl")) {
                        cl_mem *temp = (cl_mem *)va_arg(ap, cl_mem*);
                        err = clSetKernelArg(mKernel, num, sizeof(cl_mem), temp);
                        checkErrorStatus(err, "Unable to set argument");
                }
                ptr = strtok(NULL, " ");
                num += 1;
        }

        va_end(ap);
}
