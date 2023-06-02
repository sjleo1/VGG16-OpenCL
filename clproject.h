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
extern void printBuildError(cl_program, cl_device_id, cl_int);