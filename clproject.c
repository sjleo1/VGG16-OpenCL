#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "clproject.h"
#include "customlib.h"

void printDeviceInfo() {
	cl_int err_num;

	// Get number of platforms
	cl_uint num_platforms;
	err_num = clGetPlatformIDs(0, NULL, &num_platforms);
	CHECK_ERROR(err_num);

	// Get platform IDs
	cl_platform_id* platforms = (cl_platform_id*)malloc_c(sizeof(cl_platform_id) * num_platforms);
	err_num = clGetPlatformIDs(num_platforms, platforms, NULL);
	CHECK_ERROR(err_num);

	printf("Number of Platforms\t.\t.\t.\t%i\n", num_platforms);

	for (unsigned int p = 0; p < num_platforms; ++p) {
		char str[1024];

		printf("PLATFORM %i:\n", p);

		// Platform name
		err_num = clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, 1024, str, NULL);
		printf("    NAME\t.\t.\t.\t.\t%s\n", str);

		// Platform vendor
		err_num |= clGetPlatformInfo(platforms[p], CL_PLATFORM_VENDOR, 1024, str, NULL);
		printf("    VENDOR\t.\t.\t.\t.\t%s\n", str);

		// Number of devices in the platform
		cl_uint num_devices;
		err_num |= clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
		printf("    Number of Devices\t.\t.\t.\t%i\n", num_devices);

		// Get device IDs
		cl_device_id* devices = (cl_device_id*)malloc_c(sizeof(cl_device_id) * num_devices);
		err_num |= clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

		CHECK_ERROR(err_num);

		for (unsigned int d = 0; d < num_devices; ++d) {
			printf("    DEVICE %i:\n", d);

			// Device type
			cl_device_type device_type;
			err_num = clGetDeviceInfo(devices[d], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
			printf("        TYPE\t.\t.\t.\t.\t");
			if (device_type & CL_DEVICE_TYPE_CPU)
				printf("CL_DEVICE_TYPE_CPU ");
			if (device_type & CL_DEVICE_TYPE_GPU)
				printf("CL_DEVICE_TYPE_GPU ");
			if (device_type & CL_DEVICE_TYPE_ACCELERATOR)
				printf("CL_DEVICE_TYPE_ACCELERATOR ");
			if (device_type & CL_DEVICE_TYPE_DEFAULT)
				printf("CL_DEVICE_TYPE_DEFAULT ");
			if (device_type & CL_DEVICE_TYPE_CUSTOM)
				printf("CL_DEVICE_TYPE_CUSTOM ");
			printf("\n");

			// Device name
			err_num |= clGetDeviceInfo(devices[d], CL_DEVICE_NAME, 1024, str, NULL);
			printf("        NAME\t.\t.\t.\t.\t%s\n", str);

			// Device vendor
			err_num |= clGetDeviceInfo(devices[d], CL_DEVICE_VENDOR, 1024, str, NULL);
			printf("        VENDOR\t.\t.\t.\t.\t%s\n", str);

			// Device version
			err_num |= clGetDeviceInfo(devices[d], CL_DEVICE_VERSION, 1024, str, NULL);
			printf("        VERSION\t.\t.\t.\t.\t%s\n", str);

			// Device maximum clock frequency
			cl_uint max_clock_frequency;
			err_num |= clGetDeviceInfo(devices[d], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &max_clock_frequency, NULL);
			printf("        MAX_CLOCK_FREQUENCY\t.\t.\t%i MHz\n", max_clock_frequency);

			// Device maximum compute units
			cl_uint max_compute_units;
			err_num |= clGetDeviceInfo(devices[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &max_compute_units, NULL);
			printf("        MAX_COMPUTE_UNITS\t.\t.\t%i\n", max_compute_units);

			// Device maximum work item dimensions
			cl_uint max_work_item_dimensions;
			err_num |= clGetDeviceInfo(devices[d], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &max_work_item_dimensions, NULL);
			printf("        MAX_WORK_ITEM_DIMENSIONS\t.\t%i\n", max_work_item_dimensions);

			// Device maximum work item sizes
			size_t* max_work_item_sizes = (size_t*)malloc_c(sizeof(size_t) * max_work_item_dimensions);
			err_num |= clGetDeviceInfo(devices[d], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * max_work_item_dimensions, max_work_item_sizes, NULL);
			printf("        MAX_WORK_ITEM_SIZES\t.\t.\t");
			for (unsigned int i = 0; i < max_work_item_dimensions; ++i)
				printf("%zu ", max_work_item_sizes[i]);
			printf("\n");
			free_c(max_work_item_sizes);

			// Device maximum work group size
			size_t max_workgroup_size;
			err_num |= clGetDeviceInfo(devices[d], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_workgroup_size, NULL);
			printf("        MAX_WORK_GROUP_SIZE\t.\t.\t%zu\n", max_workgroup_size);

			// Device global memory size
			cl_ulong global_mem_size;
			err_num |= clGetDeviceInfo(devices[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, NULL);
			printf("        GLOBAL_MEM_SIZE\t.\t.\t.\t%llu GB\n", global_mem_size >> 30);

			// Device local memory size
			cl_ulong local_mem_size;
			err_num |= clGetDeviceInfo(devices[d], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, NULL);
			printf("        LOCAL_MEM_SIZE\t.\t.\t.\t%llu KB\n", local_mem_size >> 10);

			// Device queue properties
			cl_ulong queue_properties;
			err_num |= clGetDeviceInfo(devices[d], CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_ulong), &queue_properties, NULL);
			printf("        QUEUE_PROPERTIES\t.\t.\t");
			if (queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
				printf("CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE ");
			if (queue_properties & CL_QUEUE_PROFILING_ENABLE)
				printf("CL_QUEUE_PROFILING_ENABLE ");
			printf("\n");

			CHECK_ERROR(err_num);
		}

		free_c(devices);
	}

	free_c(platforms);

	return;
}

char* getKernelSourceCode(const char* file_name, size_t* source_length) {
	FILE* stream = fopen_c(file_name, "r");

	fseek(stream, 0, SEEK_END);
	*source_length = (size_t)ftell(stream);
	rewind(stream);

	char* source_code = (char*)malloc_c(sizeof(char) * (*source_length + 1));
	fread_c(source_code, *source_length, 1, stream);

	for (unsigned int i = 0; i < *source_length; ++i)
		if (source_code[i] == '\n')
			--* source_length;
	source_code[*source_length] = '\0';

	fclose_c(stream);

	return source_code;
}

void printBuildError(cl_program program, cl_device_id device, cl_int build_err_num) {
	if (build_err_num != CL_BUILD_PROGRAM_FAILURE) return;

	cl_int err_num;

	size_t log_size;
	err_num = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
	CHECK_ERROR(err_num);

	char* log = (char*)malloc_c(log_size + 1);
	err_num = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
	CHECK_ERROR(err_num);

	log[log_size] = '\0';
	fprintf(stderr, "OpenCL compiler error:\n%s", log);

	free_c(log);

	exit(EXIT_FAILURE);
}
