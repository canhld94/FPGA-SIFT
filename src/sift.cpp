#include "sift.h"
#include "myutil.h"
#include <stdio.h>

cl_float* readImgtoFloat(const char* inputFileName, int* width, int* height){

	cv::Mat image = cv::imread(inputFileName);
	if(!image.data){
		std::cout << "[ERROR] fail to read input image";
		exit(0);
	}

	*width = image.cols;
	*height = image.rows;

	// Convert Int img to Float img
	cv::Mat tmpimg;
	image.convertTo(tmpimg, CV_8UC3);
	cl_float *img = new cl_float[(*width)*(*height)];

	// rgb2float
	rgb2float(image.ptr<cl_uchar>(), img, (*width), (*height));

	return img;
}

void prepareFilter(std::map<int, float*>& gaussianFilters, std::map<int, cl_mem>& gaussianFiltersToDevice, oclHardware hardware){

	// setting of scales by following [1].
	// but, we only use first 5 scales descripted in [1].
	//[1] "High-Poerformance SIFT Hardware Accelerator for Real-Time Image Feature Extraction", Feng-Chen et al., Section II.A 
	int flen[SCALES] = {9, 11, 13, 15, 21};
	float sigma = INIT_SIGMA; // Initialize sigma
	float* _gaussianFilter;
	cl_int err = 0;

	for (int i=0 ; i<SCALES; i++){
	  int idx = 0;
	  int _flen = flen[i];
	  _gaussianFilter = new float[_flen*_flen];

	  for (int x=-_flen/2 ; x<=_flen/2; x++){
	  	  	 for(int y=-_flen/2; y<=_flen/2; y++){
	  	  		 float term = x*x + y*y;
	  	  		 term = term / (2*sigma*sigma);
	  	  		 _gaussianFilter[idx] = exp(-term) / (2 * 3.14f * sigma * sigma);
	  	  		 idx++;
	  	  	 }
	    }

	 gaussianFilters[i] =  _gaussianFilter;
	 gaussianFiltersToDevice[i] = clCreateBuffer(hardware.mContext,
						     CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
						     (_flen*_flen)*sizeof(float),
						     gaussianFilters[i],
						     &err);
	 sigma = sigma * 1.25992105; // sigma *= k where k is 2^(1/3)
	}
}
