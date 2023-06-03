#pragma once

#define INFO_SIZE 1024

#ifdef _MSC_VER
#pragma warning (disable: 4996)
#endif

#define CHECK_ERROR(err_num) \
	if (err_num != CL_SUCCESS) { \
		fprintf(stderr, "%s(%ld)", __FILE__, __LINE__); \
		exit(err_num); \
	}

extern void printDeviceInfo();
extern char* getKernelSourceCode(const char*, size_t*);
extern void printBuildError(const cl_program, const cl_uint, const cl_device_id*, const cl_int);
extern cl_device_id* selectPlatformAndDevices(cl_platform_id*, cl_uint*);
extern cl_program buildCLProgram(cl_context, cl_uint, const char**, cl_uint, const cl_device_id*, const char*);
