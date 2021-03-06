/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//
// Copyright (C) 2018, Network and Computing Laboratory, KAIST, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/


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
