/*
 * sift.hpp
 *
 *  Created on: May 18, 2018
 *      Author: canhld
 */

#ifndef SIFT_HPP_
#define SIFT_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/hal/hal.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types_c.h>
#include <iostream>
#include <sys/time.h>

#include <omp.h>
#include "xcl2.hpp"

using namespace cv;

// data type
typedef short pixel_t;

// extern from main.cpp
extern cl::Context context;
//extern std::vector<cl::CommandQueue> q;
extern cl::CommandQueue q;
extern std::vector<cl::Kernel> gaussian;
extern cl::Buffer base;
extern std::vector<cl::Buffer> gpyr_dev;

// extern from sift.cpp
extern const int nScales;
extern const int nOctaves;
extern std::vector<Mat> gpyr;

// Image size
#define COLS 1024
#define ROWS 1024

#define DATATYPE CV_16SC1

 /*SIFT build-in opencv function*/
void SITF_BuildIn_OpenCV(InputArray image,
						 std::vector<KeyPoint>& keypoints,
						 OutputArray descriptors);

void SIFT_NCL_CPU(InputArray image,
		  std::vector<KeyPoint> & keypoints,
		  OutputArray descriptors);

 /*NCL SIFT, based opencv source code*/
void SIFT_NCL(InputArray image,
		  std::vector<KeyPoint> & keypoints,
		  OutputArray descriptors);

/*Sub modules*/

void Gaussian_Blur(Mat& src, Mat& dst, double sigma);

void Gaussian_Blur_1D(Mat& src, Mat& dst, double sigma);

void buildGaussianPyramid(Mat& image);

void buildDoGPyramid(std::vector<Mat>& aGpyr,
					 std::vector<Mat>& dogpyr);

void findScaleSpaceExtrema(std::vector<Mat>& aGpyr,
						   std::vector<Mat>& dogpyr,
						   std::vector<KeyPoint>& keypoints);

void calDescriptor( std::vector<Mat>& aGpyr,
					std::vector<KeyPoint>& keypoints,
					Mat& descriptors,
					int firstOctave);


#endif /* SIFT_HPP_ */
