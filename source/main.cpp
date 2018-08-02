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


#include "../include/sift.hpp"

// Definition of gaussian pyramid and dog pyramid


// Definition of Opencl mess
cl::Device device;
cl::Context context;
cl::CommandQueue q; // use 1 out of oder queue
//std::vector<cl::CommandQueue> q; // 1 for htod, 1 for dtoh, 1 for kernel execution
cl::Program program;
std::vector<cl::Kernel> gaussian(nScales);

// Definition of memory object in FPGA
cl::Buffer base;
std::vector<cl::Buffer> gpyr_dev;

void fpgaConfig(){
    // find Xilinx device and program FPGA
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    device = devices[0];
    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::cout << "Found Device = " << device_name.c_str() << std::endl;
	std::string binaryFile = "sift.xclbin";
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    
    std::cout << "Imported binary" << std::endl;
    // create context and command queue
    context = cl::Context(device);
    q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
    // creat program and kernel
    devices.resize(1);
    program = cl::Program(context, devices, bins);
    gaussian = {cl::Kernel(program,"gaussian_s0"), cl::Kernel(program,"gaussian_s1"),\
                cl::Kernel(program,"gaussian_s2"), cl::Kernel(program,"gaussian_s3"),\
                cl::Kernel(program,"gaussian_s4") };
    std::cout << "FPGA config done" << std::endl;

}

void hostPreAllocation(){
    gpyr.resize(nOctaves*nScales);
    for(int o = 0; o < nOctaves; ++o){
        for(int i = 0; i < nScales; ++i){
            Mat &img = gpyr[o*nScales + i];
        	img = cv::Mat(ROWS >> o, COLS >> o, CV_16SC1);
        }
    }
}


void fpgaPreAllocation(){
    // opencl extern flag use mutiple ddr bank
    cl_mem_ext_ptr_t inExt;
    std::vector<cl_mem_ext_ptr_t> outExt(4); // 4 ddr bank
    inExt.flags = (1 << 8); // DDR BANK 0
    inExt.obj = 0;
    inExt.param = 0;
    for( int i = 0; i < 4; ++i){
        outExt[i].flags = (1 << (8+i)); // DDR bank i
    }
    // alocate base image
    int size_in_bytes = ROWS*COLS*sizeof(pixel_t);
    base = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, size_in_bytes, &inExt);
    for(int i = 0; i < nScales; ++i){
    	int bank = i % 4;
    	gpyr_dev.push_back(cl::Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX, size_in_bytes, &outExt[bank]));
    }
}
void setKernelArg(){
	for(int i = 0; i < nScales; ++i){
		int narg = 0;
		gaussian[i].setArg(narg++, base);
		if(i == 0) {
			gaussian[i].setArg(narg++, base);
			gaussian[i].setArg(narg++, ROWS);
			gaussian[i].setArg(narg++, COLS);
			gaussian[i].setArg(narg++, 0);
		}
		else gaussian[i].setArg(narg++, gpyr_dev[i]);
	}
}

void releaseApplication(){
	// delete host memory
	gpyr.clear();
	// release all OpenCL object
	// release memory buffer
	gpyr_dev.clear();
	base.~Wrapper();
	// release kernel
	gaussian.clear();
	// release command queue
	q.~Wrapper();
	// release program
	program.~Wrapper();
	// release context
	context.~Wrapper();
	// release device
	device.~Wrapper();
}

void readme(){
	std::cout<< "Usage: ./sift <scene> <object> " << std::endl;
}

void readImage(Mat& img, Mat& gray, char* filename, bool resized){
	img = imread(filename);
	if(!img.data)
	{ std::cout<< " --(!) Error reading images " << std::endl; exit(0); }
	if(resized) resize(img, img, Size(ROWS,COLS));
	cvtColor(img, gray, cv::COLOR_RGB2GRAY);
	gray.convertTo(gray, DATATYPE);
	return;
}

int main( int argc, char** argv )
{
    if( argc != 3 )
    { readme(); return -1; }
    hostPreAllocation();
    fpgaConfig();
    fpgaPreAllocation();
    setKernelArg();

    std::cout << "Start computing" << std::endl;

    Mat img0, img1; // image
    Mat gray0, gray1;
    std::vector<KeyPoint> keypoints0, keypoints1; // keypoints to store keypoints
    Mat descriptors0, descriptors1; // image descriptor
    readImage(img0, gray0 , argv[1], 1);
    readImage(img1, gray1, argv[2], 0);
    SIFT_NCL_CPU(gray1, keypoints1, descriptors1);
    SIFT_NCL(gray0, keypoints0, descriptors0);
    BFMatcher matcher(NORM_L2);
    std::vector<std::vector<DMatch> > matches;
    matcher.knnMatch(descriptors1, descriptors0, matches, 2);
    std::vector<DMatch> good_matches;
    good_matches.reserve(matches.size());
    for (size_t i = 0; i < matches.size(); ++i){
        if (matches[i].size() < 2)
                    continue;

        const DMatch &m1 = matches[i][0];
        const DMatch &m2 = matches[i][1];

        if(m1.distance <= 0.8*m2.distance)
        good_matches.push_back(m1);
    }
    Mat img_matches;
    drawMatches(img1, keypoints1, img0, keypoints0, good_matches, img_matches, Scalar(0, 0, 255), Scalar::all(-1),std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    std::vector< Point2f >  obj;
    std::vector< Point2f >  scene;

    for( unsigned int i = 0; i < good_matches.size(); i++ ){
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints0[ good_matches[i].trainIdx ].pt );
    }

    Mat H = findHomography( obj, scene, RANSAC );

    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector< Point2f > obj_corners(4);
    obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img1.cols, 0 );
    obj_corners[2] = cvPoint( img1.cols, img1.rows ); obj_corners[3] = cvPoint( 0, img1.rows );
    std::vector< Point2f > scene_corners(4);

    perspectiveTransform( obj_corners, scene_corners, H);

//-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line( img_matches, scene_corners[0] + Point2f( img1.cols, 0), scene_corners[1] + Point2f( img1.cols, 0), Scalar(0, 255, 0), 4 );
    line( img_matches, scene_corners[1] + Point2f( img1.cols, 0), scene_corners[2] + Point2f( img1.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[2] + Point2f( img1.cols, 0), scene_corners[3] + Point2f( img1.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[3] + Point2f( img1.cols, 0), scene_corners[0] + Point2f( img1.cols, 0), Scalar( 0, 255, 0), 4 );
    imwrite("matched.jpg", img_matches);
    std::cout << "done!" << std::endl;
	releaseApplication();
    return 0;
}
