#ifndef __OCL_HELPER_H__
#define __OCL_HELPER_H__

#include <CL/cl.h>

struct oclHardware {
    cl_platform_id mPlatform;
    cl_context mContext;
    cl_device_id mDevice;
    cl_command_queue mQueue;
    cl_command_queue IOQueue;
    short mMajorVersion;
    short mMinorVersion;
};

struct oclSoftware {

	cl_program mProgram;
	cl_kernel mKernel;
	char mKernelName[128];
	char mFileName[1024];
	char mCompileOptions[1024];
};

oclHardware getOclHardware(cl_device_type type, const char *target_device);

int getOclSoftware(oclSoftware &software, const oclHardware &hardware);

void release(oclSoftware& software);

void release(oclHardware& hardware);

const char *oclErrorCode(cl_int code);


#endif