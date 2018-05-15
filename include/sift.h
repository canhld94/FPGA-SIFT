#ifndef __SIFT_H__
#define __SIFT_H__

#include <CL/cl.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "bitmap.h"
#include "oclHelper.h"

#define SCALES 5
#define INIT_SIGMA 1.6

cl_float* readImgtoFloat(const char* inputFileName, int* width, int* height);
void prepareFilter(std::map<int, float*>& gaussianFilters, std::map<int, cl_mem>& gaussianFiltersToDevice, oclHardware hardware);

#endif
