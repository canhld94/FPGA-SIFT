/********************************************
FPGA-SIFT using OpenCL

:Network Computing Lab: (ncl.kaist.ac.kr)
:last_update:	2018-05-15
**********************************************/

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include "myutil.h"
#include "oclHelper.h"
#include "bitmap.h"
#include "sift.h"

int main(int argc, char* argv[]){

	int width, height = 128; //default img width height
	cl_float *img;
	oclHardware hardware;
	oclSoftware software;
	cl_kernel  mKernel_InputToOnchip,
	     mKernel_FilterToOnchip_scale_0,
	     mKernel_FilterToOnchip_scale_1,
	     mKernel_FilterToOnchip_scale_2,
	     mKernel_FilterToOnchip_scale_3,
	     mKernel_FilterToOnchip_scale_4,
	     mKernel_Shift_reg_begin,
	     mKernel_Shift_reg_loop,
	     mKernel_Shift_reg_end,
	     mKernel_conv_oct,
	     mKernel_dog_oct,
	     mKernel_OutputFromOnchip;
	int flen[SCALES] = {9, 11, 13, 15, 21};

	// TARGET_DEVICE macro needs to be passed from gcc command line
	const char * target_device_name = TARGET_DEVICE;
	std::cout << target_device_name;

	// Running application exception handling
	if (argc != 3){
		std::cout 	<< "Usage " 
					<< argv[0] 
					<< " input bitmap> <xclbin> \n";
		return -1;
	}

	// Read the bit map file into memory and allocate memory for the final image
	std::cout 	<< "\n***************\n" 
				<< "Reading input imeage ... \n" 
				<< "***************\n";
	const char* inputFileName = argv[1];
	const char* xclbinFilename = argv[2];
	img = readImgtoFloat(inputFileName,&width, &height);


	// Set up OpenCL hardware and software contructs
	std::cout << "\n***************\n" ;
	std::cout << "Setting up OpenCL hardware and software ... \n" << "***************\n";
	cl_int err = 0;

	hardware  = getOclHardware(CL_DEVICE_TYPE_ACCELERATOR, target_device_name);
	std::cout << hardware.mContext;

	memset(&software, 0, sizeof(oclSoftware));
	strcpy(software.mFileName, xclbinFilename);
	strcpy(software.mCompileOptions, "-g -Wall");

	getOclSoftware(software,hardware);

	// Create Kernels
	clCreateKernel_withErrorCheck(mKernel_InputToOnchip, software, "InputToOnchip");
	clCreateKernel_withErrorCheck(mKernel_FilterToOnchip_scale_0, software, "FilterToOnchip_scale_0");
	clCreateKernel_withErrorCheck(mKernel_FilterToOnchip_scale_1, software, "FilterToOnchip_scale_1");
	clCreateKernel_withErrorCheck(mKernel_FilterToOnchip_scale_2, software, "FilterToOnchip_scale_2");
	clCreateKernel_withErrorCheck(mKernel_FilterToOnchip_scale_3, software, "FilterToOnchip_scale_3");
	clCreateKernel_withErrorCheck(mKernel_FilterToOnchip_scale_4, software, "FilterToOnchip_scale_4");
	clCreateKernel_withErrorCheck(mKernel_Shift_reg_begin, software, "Shift_reg_begin");
	clCreateKernel_withErrorCheck(mKernel_Shift_reg_loop, software, "Shift_reg_loop");
	clCreateKernel_withErrorCheck(mKernel_Shift_reg_end, software, "Shift_reg_end");
	clCreateKernel_withErrorCheck(mKernel_conv_oct, software, "conv_oct");
	clCreateKernel_withErrorCheck(mKernel_dog_oct, software, "dog_oct");
	clCreateKernel_withErrorCheck(mKernel_OutputFromOnchip, software, "OutputFromOnchip");

	// Initialize OpenCL buffers with pointers to allocated memory
	cl_mem imageToDevice;
	cl_mem imageFromDevice;
	cl_mem DoGsFromDevice;
	std::map<int, cl_mem> gaussianFiltersToDevice;
	std::map<int, float*> gaussianFilters;
	float *DoGPyramid = new float[SCALES*height*width];

	// prepare
	prepareFilter(gaussianFilters, gaussianFiltersToDevice, hardware);


	// Create Buffer Image to Device Buffer
	imageToDevice = clCreateBuffer(hardware.mContext,
		CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
		width * height * sizeof(float),
		img,
		&err);
	checkErrorStatus(err, "Unable to create write buffer");

	// Create Buffer DoG From Device Buffer
	DoGsFromDevice = clCreateBuffer(hardware.mContext,
				   CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
				   SCALES * width * height * sizeof(float),
				   DoGPyramid,
				   &err) ;
	checkErrorStatus(err, "Unable to create write buffer") ;

	// Send the image to the hardware
	std::cout << "Writing input image to buffer ... \n";
	err = clEnqueueWriteBuffer(hardware.mQueue,
		imageToDevice,
		CL_TRUE,
		0,
		width * height * sizeof(int),
		img,
		0,
		NULL,
		NULL);
	checkErrorStatus(err, "Unable to enqueue write buffer");

  	// Send the gaussian filters to the hardware
  	std::cout << "Writing filter to buffer...\n";
  	for(int i=0; i<SCALES; i++){
		int _flen = flen[i];
		err = clEnqueueWriteBuffer(hardware.IOQueue, gaussianFiltersToDevice[i], CL_TRUE, 0, _flen*_flen*sizeof(float), gaussianFilters[i], 0, NULL, NULL);
		checkErrorStatus(err, "Unable to enqueue write buffer : gaussianFiltersToDevice");
	}

	// Pass the arguments to the kernel
  	char * temp_string;
  	temp_string = new char[129];
  	printf("Set argument for InputToOnchip kernel...\n");
  	strcpy(temp_string, "%cl");
  	setArguments(mKernel_InputToOnchip, temp_string, &imageToDevice);
	
  	printf("Set argument for FilterToOnchip_scale_0 kernel...\n");
  	strcpy(temp_string, "%cl");
  	setArguments(mKernel_FilterToOnchip_scale_0, temp_string, &gaussianFiltersToDevice[0]);

  	printf("Set argument for FilterToOnchip_sclae_1 kernel...\n");
  	strcpy(temp_string, "%cl");
  	setArguments(mKernel_FilterToOnchip_scale_1, temp_string, &gaussianFiltersToDevice[1]);
	
  	printf("Set argument for FilterToOnchip_scale_2 kernel...\n");
  	strcpy(temp_string, "%cl");
  	setArguments(mKernel_FilterToOnchip_scale_2, temp_string, &gaussianFiltersToDevice[2]);
	
  	printf("Set argument for FilterToOnchip_scale_3 kernel...\n");
  	strcpy(temp_string, "%cl");
  	setArguments(mKernel_FilterToOnchip_scale_3, temp_string, &gaussianFiltersToDevice[3]);
	
  	printf("Set argument for FilterToOnchip_scale_4 kernel...\n");
  	strcpy(temp_string, "%cl");
  	setArguments(mKernel_FilterToOnchip_scale_4, temp_string, &gaussianFiltersToDevice[4]);
	
  	printf("Set argument for OutputFromOnchip kernel...\n");
  	strcpy(temp_string, "%cl");
  	setArguments(mKernel_OutputFromOnchip, temp_string, &DoGsFromDevice);
	
  	// Define iteration space 
  	size_t globalSize[3] = { 1, 1, 1 } ;
  	size_t localSize[3] = { 1, 1, 1} ;
  	cl_event seq_complete ;
	
  	// Actually start the kernels on the hardware
  	printf("Start the InputToOnchip kernel...\n");
  	err = clEnqueueNDRangeKernel(hardware.mQueue,
					mKernel_InputToOnchip,
					1,
					NULL,
					globalSize,
					localSize,
					0,
					NULL,
					&seq_complete);
  	checkErrorStatus(err, "Unable to enqueue InputToOnchip...\n");
	
  	printf("Start the FilterToOnchip kernel...\n");
  	err = clEnqueueNDRangeKernel(hardware.mQueue,
			       	mKernel_FilterToOnchip_scale_0,
			       	1,
			       	NULL,
			       	globalSize,
			       	localSize,
			       	0,
			       	NULL,
			       	&seq_complete) ;
  	checkErrorStatus(err, "Unable to enqueue FilterToOnchip_scale_0...\n") ;
	
  	printf("Start the FilterToOnchip_scale_1 kernel ...\n");
  	err = clEnqueueNDRangeKernel(hardware.mQueue,
					mKernel_FilterToOnchip_scale_1,
					1,
					NULL,
					globalSize,
					localSize,
					0,
					NULL,
					&seq_complete);
  	checkErrorStatus(err, "Unable to enqueue FilterToOnchip_scale_1...\n");
	
  	printf("Start the Shift_reg_begin kernel ...\n");
  	err = clEnqueueNDRangeKernel(hardware.mQueue,
					mKernel_Shift_reg_begin,
					1,
					NULL,
					globalSize,
					localSize,
					0,
					NULL,
					&seq_complete);
  	checkErrorStatus(err, "Unable to enqueue Shift_reg_begin...\n");
 	
  	printf("Start the Shift_reg_loop kernel ...\n");
  	err = clEnqueueNDRangeKernel(hardware.mQueue,
					mKernel_Shift_reg_loop,
					1,
					NULL,
					globalSize,
					localSize,
					0,
					NULL,
					&seq_complete);
  	checkErrorStatus(err, "Unable to enqueue Shift_reg_loop...\n");
	
  	printf("Start the Shift_reg_end kernel ...\n");
  	err = clEnqueueNDRangeKernel(hardware.mQueue,
					mKernel_Shift_reg_end,
					1,
					NULL,
					globalSize,
					localSize,
					0,
					NULL,
					&seq_complete);
  	checkErrorStatus(err, "Unable to enqueue Shift_reg_end...\n");
	
  	printf("Start the conv_oct kernel ...\n");
  	err = clEnqueueNDRangeKernel(hardware.mQueue,
					mKernel_conv_oct,
					1,
					NULL,
					globalSize,
					localSize,
					0,
					NULL,
					&seq_complete);
  	checkErrorStatus(err, "Unable to enqueue conv_oct...\n");
	
  	printf("Start the dog_oct kernel ...\n");
  	err = clEnqueueNDRangeKernel(hardware.mQueue,
					mKernel_dog_oct,
					1,
					NULL,
					globalSize,
					localSize,
					0,
					NULL,
					&seq_complete);
  	checkErrorStatus(err, "Unable to enqueue dog_oct...\n");
	
  	printf("Start the OutputFromOnchip kernel ...\n");
  	err = clEnqueueNDRangeKernel(hardware.mQueue,
					mKernel_OutputFromOnchip,
					1,
					NULL,
					globalSize,
					localSize,
					0,
					NULL,
					&seq_complete);
  	checkErrorStatus(err, "Unable to enqueue OutputFromOnchip...\n");
	
	
  	// Wait for kernel to finish
  	// Read back the image from the kernel
  	std::cout << "Reading output DoG...\n";
  	err = clEnqueueReadBuffer(hardware.IOQueue,
			    	DoGsFromDevice,
			    	CL_TRUE,
			    	0,
			    	SCALES * width * height * sizeof(float),
			    	DoGPyramid,
			    	0,
			    	NULL,
			    	&seq_complete) ;
	
  	checkErrorStatus(err, "Unable to enqueue read buffer") ;
	
  	clWaitForEvents(1, &seq_complete) ;
	
  	// Write the final image to disk
  	//image.writeBitmapFile(outImage) ;
	
  	for (int i=0; i<SCALES; i++){
	  	clReleaseMemObject(gaussianFiltersToDevice[i]);
  	}
  	clReleaseMemObject(imageToDevice);
  	clReleaseMemObject(DoGsFromDevice);
  	
  	clReleaseKernel(mKernel_InputToOnchip);
  	clReleaseKernel(mKernel_FilterToOnchip_scale_0);
  	clReleaseKernel(mKernel_FilterToOnchip_scale_1);
  	clReleaseKernel(mKernel_FilterToOnchip_scale_2);
  	clReleaseKernel(mKernel_FilterToOnchip_scale_3);
  	clReleaseKernel(mKernel_FilterToOnchip_scale_4);
  	clReleaseKernel(mKernel_Shift_reg_begin);
  	clReleaseKernel(mKernel_Shift_reg_loop);
  	clReleaseKernel(mKernel_Shift_reg_end);
  	clReleaseKernel(mKernel_conv_oct);
  	clReleaseKernel(mKernel_dog_oct);
  	clReleaseKernel(mKernel_OutputFromOnchip);
  	release(software) ;
  	release(hardware) ;
  	return 0 ;
}
