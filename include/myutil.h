#ifndef __MYUTIL_H__
#define __MYUTIL_H__

#include <iostream>
#include <fstream>
#include <vector>
#include <stdarg.h>
#include <CL/cl.h>
#include <stdio.h>
#include <string>
#include "oclHelper.h"

#if defined(SDX_PLATFORM) && !defined(TARGET_DEVICE)
	#define STR_VALUE(arg)	#arg
	#define GET_STRING(name) STR_VALUE(name)
	#define TARGET_DEVICE GET_STRING(SDX_PLATFORM)
#endif

template <typename T>
struct aligned_allocator{
	using value_type = T;
	T* allocate(std::size_t num){
		void* ptr = nullptr;
		if (posix_memalign(&ptr,4096,num*sizeof(T)))
			throw std::bad_alloc();
		return reinterpret_cast<T*>(ptr);
	}
	void deallocate(T* p, std::size_t num){
		free(p);
	}
};

void checkErrorStatus(cl_int error, const char* message);
void rgb2float(cl_uchar* srcImg, cl_float *(&dstImg), int width, int height);
void clCreateKernel_withErrorCheck(cl_kernel& dstKernelVar, oclSoftware& software, char* kernel);
void setArguments(cl_kernel mKernel, char *input_type, ...);

#endif
