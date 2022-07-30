#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cnn.h"

extern const char* CLASS_NAME[];
extern const int INPUT_DIM[];
extern const int OUTPUT_DIM[];
extern const int NBYN[];

void parallelCNNInit();
void parallelCNN(float* image, float** networks, int* labels, float* confidences, int num_images);
void showInfo();
char* getSourceCode(const char* file_name, size_t* len);
void buildError(cl_program program, cl_device_id device, cl_int err);
// conv, maxp, fcl, softmax functions here...


void parallelCNN(float* image, float** networks, int* labels, float* confidences, int num_images) {
	cl_int err;

	showInfo();

	// Initialization	
	// Selecting platform
	cl_uint platform_num, num_platforms;
	printf("Platform number?\n");
	scanf("%d", &platform_num);
	err = clGetPlatformIDs(0, NULL, &num_platforms);
	CHECK_ERROR(err);
	if (platform_num >= num_platforms) {
		printf("Invalid platform number.\n");
		exit(EXIT_FAILURE);
	}
	cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
	err = clGetPlatformIDs(num_platforms, platforms, NULL);
	CHECK_ERROR(err);

	// Selecting device
	cl_uint device_num, num_devices;
	printf("Device number?\n");
	scanf("%d", &device_num);
	err = clGetDeviceIDs(platforms[platform_num], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
	CHECK_ERROR(err);
	if (device_num >= num_devices) {
		printf("Invalid device number.\n");
		exit(EXIT_FAILURE);
	}
	cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
	err = clGetDeviceIDs(platforms[platform_num], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
	CHECK_ERROR(err);

	// Creating context
	cl_context context = clCreateContext(NULL, 1, &devices[device_num], NULL, NULL, &err);
	CHECK_ERROR(err);

	// Creating command queue
	cl_command_queue queue = clCreateCommandQueue(context, devices[device_num], CL_QUEUE_PROFILING_ENABLE, &err);
	CHECK_ERROR(err);

	// Creating a program object
	size_t kernel_source_size;
	const char* kernel_source = getSourceCode("cnn_kernel.cl", &kernel_source_size);
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
	CHECK_ERROR(err);

	// Compiling and linking a program with build options
	char build_option[128];
	sprintf(build_option, "-cl-fast-relaxed-math -D DEPTH=%d -D BATCH..."/*, DEPTH, etc...*/);
	err = clBuildProgram(program, 1, &devices[device_num], build_option, NULL, NULL);
	CHECK_ERROR(err);
	buildError(program, devices[device_num], err);

	// Creating kernels
	cl_kernel test_kernel = clCreateKernel(program, "convolution_ker", &err);
	CHECK_ERROR(err);








	// Computation
	time_t start = clock();
	
	//cl_mem* images_mem = (cl_mem*)malloc(sizeof(cl_mem) * num_images);
	cl_mem* filters_mem = (cl_mem*)malloc(sizeof(cl_mem) * 16);
	cl_mem* biases_mem = (cl_mem*)malloc(sizeof(cl_mem) * 16);
	cl_mem* labels_mem = (cl_mem*)malloc(sizeof(cl_mem) * num_images);
	cl_mem* confidences_mem = (cl_mem*)malloc(sizeof(cl_mem) * num_images);



	for (int s = 0, m = 0, n = 0; s < 21; s++) {
		// Max pooling layer doesn't need filters and biases
		if (s == 2 || s == 5 || s == 9 || s == 13 || s == 17)
			continue;
		// For convolution layers
		else if (s < 18) {
			filters_mem[m] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 9 * INPUT_DIM[s] * OUTPUT_DIM[s], NULL, &err);
			err |= clEnqueueWriteBuffer(queue, filters_mem[m], CL_FALSE, 0, sizeof(float) * 9 * INPUT_DIM[s], OUTPUT_DIM[s], networks[n++], 0, NULL, NULL);
			CHECK_ERROR(err);

			biases_mem[m] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * INPUT_DIM[s] * OUTPUT_DIM[s], NULL, &err);
			err |= clEnqueueWriteBuffer(queue, biases_mem[m], CL_FALSE, 0, sizeof(float) * INPUT_DIM[s] * OUTPUT_DIM[s], networks[n++], 0, NULL, NULL);
			CHECK_ERROR(err);

			m++;
		}
		// For fully connected layers
		else {
			filters_mem[m] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * INPUT_DIM[s] * OUTPUT_DIM[s], NULL, &err);
			err |= clEnqueueWriteBuffer(queue, filters_mem[m], CL_FALSE, 0, sizeof(float) * OUTPUT_DIM[s], networks[n++], 0, NULL, NULL);
			CHECK_ERROR(err);

			biases_mem[m] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * OUTPUT_DIM[s], NULL, &err);
			err |= clEnqueueWriteBuffer(queue, biases_mem[m], CL_FALSE, 0, sizeof(float) * OUTPUT_DIM[s], networks[n++], 0, NULL, NULL);
			CHECK_ERROR(err);

			m++;
		}
	}

	//cl_mem* temp_mem_A = clCreateBuffer(context, CL_MEM_READ_WRITE, 65536, NULL, &err);
	//cl_mem* temp_mem_B = clCreateBuffer(context, CL_MEM_READ_WRITE, 65536, NULL, &err);
	cl_mem temp_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * OUTPUT_DIM[0] * NBYN[0] * NBYN[0], NULL, &err);
	CHECK_ERROR(err);

	cl_mem images_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 32 * 32 * 3, NULL, &err);
	err |= clEnqueueWriteBuffer(queue, images_mem, CL_FALSE, 0, sizeof(float) * 32 * 32 * 3, image, 0, NULL, NULL);
	CHECK_ERROR(err);

	convolutionCL(queue, test_kernel, &images_mem, &temp_mem, &filters_mem[0], &biases_mem[0], INPUT_DIM[0], OUTPUT_DIM[0], NBYN);

	float temp_result[32 * 32 * 64];
	clEnqueueReadBuffer(queue, temp_mem, CL_TRUE, 0, sizeof(float) * OUTPUT_DIM[0] * NBYN[0] * NBYN[0], temp_result, 0, NULL, NULL);

	time_t end = clock();
	printf("Elapsed time (OpenCL): %.2f sec.\n", (double)(end - start) / CLK_TCK);

	for (int i = 0; i < 64; i++)
		printf("%f\n", temp_result[i]);
}

void convolutionCL(cl_command_queue queue, cl_kernel kernel, cl_mem* inputs, cl_mem* outputs, cl_mem* filters, cl_mem* biases, int indim, int outdim, int N) {
	cl_int err;
	// Setting global size and local size
	size_t global_size[3] = { 32, 32, 64 };
	size_t local_size[3] = { 2, 2, 64 };
	
	// clSetKernalArg
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), inputs);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), filters);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), biases);
	err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), outputs);
	err |= clSetKernelArg(kernel, 4, sizeof(float), NULL);
	err |= clSetKernelArg(kernel, 5, sizeof(float), NULL);
	err |= clSetKernelArg(kernel, 6, sizeof(short), &indim);
	err |= clSetKernelArg(kernel, 7, sizeof(short), &outdim);
	err |= clSetKernelArg(kernel, 8, sizeof(short), &N);
	CHECK_ERROR(err);

	// clEnqueueNDRangeKernel
	err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_size, local_size, 0, NULL, NULL);
	CHECK_ERROR(err);

	return;
}









void showInfo() {
	cl_int err;
	cl_uint num_platforms, num_devices;
	cl_device_type device_type;
	size_t max_work_group_size;
	cl_ulong global_mem_size, local_mem_size, max_clock_frequency, max_compute_units, queue_properties;
	char str[1024];

	err = clGetPlatformIDs(0, NULL, &num_platforms);
	CHECK_ERROR(err);

	cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
	err = clGetPlatformIDs(num_platforms, platforms, NULL);
	CHECK_ERROR(err);

	printf("-Number of platforms\t\t\t: %u\n", num_platforms);

	for (unsigned int p = 0; p < num_platforms; p++) {
		printf("-Platform %u\n", p);

		err = clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, 1024, str, NULL);
		CHECK_ERROR(err);
		printf(" -CL_PLATFORM_NAME\t\t\t: %s\n", str);

		err = clGetPlatformInfo(platforms[p], CL_PLATFORM_VENDOR, 1024, str, NULL);
		CHECK_ERROR(err);
		printf(" -CL_PLATFORM_VENDOR\t\t\t: %s\n", str);

		err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
		CHECK_ERROR(err);
		printf(" -Number of devices\t\t\t: %u\n", num_devices);

		cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
		err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
		CHECK_ERROR(err);

		for (unsigned int d = 0; d < num_devices; d++) {
			printf(" -Device %u\n", d);

			err = clGetDeviceInfo(devices[d], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
			CHECK_ERROR(err);
			printf("  -CL_DEVICE_TYPE\t\t\t:");
			if (device_type & CL_DEVICE_TYPE_CPU) printf(" CL_DEVICE_TYPE_CPU");
			if (device_type & CL_DEVICE_TYPE_GPU) printf(" CL_DEVICE_TYPE_GPU");
			if (device_type & CL_DEVICE_TYPE_ACCELERATOR) printf(" CL_DEVICE_TYPE_ACCELERATOR");
			if (device_type & CL_DEVICE_TYPE_DEFAULT) printf(" CL_DEVICE_TYPE_DEFAULT");
			if (device_type & CL_DEVICE_TYPE_CUSTOM) printf(" CL_DEVICE_TYPE_CUSTOM");
			printf("\n");

			err = clGetDeviceInfo(devices[d], CL_DEVICE_NAME, 1024, str, NULL);
			CHECK_ERROR(err);
			printf("  -CL_DEVICE_NAME\t\t\t: %s\n", str);

			err = clGetDeviceInfo(devices[d], CL_DEVICE_VENDOR, 1024, str, NULL);
			CHECK_ERROR(err);
			printf("  -CL_DEVICE_VENDOR\t\t\t: %s\n", str);

			err = clGetDeviceInfo(devices[d], CL_DEVICE_VERSION, 1024, str, NULL);
			CHECK_ERROR(err);
			printf("  -CL_DEVICE_VERSION\t\t\t: %s\n", str);

			err = clGetDeviceInfo(devices[d], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_ulong), &max_clock_frequency, NULL);
			CHECK_ERROR(err);
			printf("  -CL_DEVICE_MAX_CLOCK_FREQUENCY\t: %lu MHz\n", max_clock_frequency);

			err = clGetDeviceInfo(devices[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_ulong), &max_compute_units, NULL);
			CHECK_ERROR(err);
			printf("  -CL_DEVICE_MAX_COMPUTE_UNITS\t\t: %lu\n", max_compute_units);

			err = clGetDeviceInfo(devices[d], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
			CHECK_ERROR(err);
			printf("  -CL_DEVICE_MAX_WORK_GROUP_SIZE\t: %lu\n", max_work_group_size);

			err = clGetDeviceInfo(devices[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, NULL);
			CHECK_ERROR(err);
			printf("  -CL_DEVICE_GLOBAL_MEM_SIZE\t\t: %lu\n", global_mem_size);

			err = clGetDeviceInfo(devices[d], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, NULL);
			CHECK_ERROR(err);
			printf("  -CL_DEVICE_LOCAL_MEM_SIZE\t\t: %llu\n", local_mem_size);

			err = clGetDeviceInfo(devices[d], CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_ulong), &queue_properties, NULL);
			CHECK_ERROR(err);
			printf("  -CL_DEVICE_QUEUE_PROPERTIES\t\t:");
			if (queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) printf(" CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE");
			if (queue_properties & CL_QUEUE_PROFILING_ENABLE) printf(" CL_QUEUE_PROFILING_ENABLE");
			printf("\n");
		}
		free(devices);
	}
	free(platforms);

	return;
}

char* getSourceCode(const char* file_name, size_t* code_length) {
	FILE* fp = fopen(file_name, "rb");
	if (fp == NULL) {
		fprintf(stderr, "Failed to open %s.\n", file_name);
		exit(EXIT_FAILURE);
	}

	fseek(fp, 0, SEEK_END);
	*code_length = (size_t)ftell(fp);
	rewind(fp);

	char* source_code = (char*)malloc(*code_length + 1);
	fread(source_code, *code_length, 1, fp);
	fclose(fp);
	source_code[*code_length] = '\0';

	return source_code;
}

void buildError(cl_program program, cl_device_id device, cl_int err) {
	if (err == CL_BUILD_PROGRAM_FAILURE) {
		size_t log_size;
		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		CHECK_ERROR(err);

		char* log = (char*)malloc(log_size + 1);
		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		CHECK_ERROR(err);

		log[log_size] = '\0';
		fprintf(stderr, "Compile error:\n%s\n", log);
		free(log);
		exit(EXIT_FAILURE);
	}

	return;
}