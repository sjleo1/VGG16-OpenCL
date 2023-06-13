#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>
#include "cnn.h"
#include "customlib.h"
#include "clproject.h"

#define BATCH 1

const size_t num_kernel_files = 4;
const char* kernel_files[] = {
	"maxp.cl",
	"conv.cl",
	"memset.cl",
	"fc.cl"
};

static inline float ReLU(float val) {
	if (val > 0.0)	return	val;
	else			return	0.0;
}

static size_t opt_work_group_length;
static size_t work_per_thread;
static void setOptimalWorkGroupSize(cl_device_id device) {
	cl_int err_num;

	size_t max_work_group_size;
	err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
	CHECK_ERROR(err_num);

	if (max_work_group_size >= 1024) {
		opt_work_group_length = 32;
		work_per_thread = 1;
	}
	else if (max_work_group_size >= 256) {
		opt_work_group_length = 16;
		work_per_thread = 2;
	}
	else {
		opt_work_group_length = 8;
		work_per_thread = 4;
	}
}

static inline void convolutionLow(
	cl_command_queue command_queue,
	cl_kernel kernel,
	cl_mem* input,
	cl_mem* output,
	cl_mem* weight,
	cl_mem* bias,
	const size_t in_width,
	const size_t out_width,
	const size_t resolution)
{
	cl_int err_num;
	cl_uint work_dim = 3;
	const size_t global_work_size[3] = { resolution / work_per_thread, resolution / work_per_thread, out_width };
	const size_t local_work_size[3] = { resolution / work_per_thread, resolution / work_per_thread, 1 };

	err_num = clSetKernelArg(kernel, 0, sizeof(cl_mem), input);
	err_num |= clSetKernelArg(kernel, 1, sizeof(cl_mem), output);
	err_num |= clSetKernelArg(kernel, 2, sizeof(cl_mem), weight);
	err_num |= clSetKernelArg(kernel, 3, sizeof(cl_mem), bias);
	err_num |= clSetKernelArg(kernel, 4, sizeof(unsigned short), &(unsigned short)in_width);
	err_num |= clSetKernelArg(kernel, 5, sizeof(unsigned short), &(unsigned short)out_width);
	err_num |= clSetKernelArg(kernel, 6, sizeof(unsigned char), &(unsigned char)resolution);
	err_num |= clSetKernelArg(kernel, 7, sizeof(float) * resolution * resolution, NULL);
	CHECK_ERROR(err_num);

	err_num = clEnqueueNDRangeKernel(command_queue, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	CHECK_ERROR(err_num);

	return;
}

static inline void convolutionHigh(
	cl_command_queue command_queue,
	cl_kernel kernel,
	cl_mem* input,
	cl_mem* output,
	cl_mem* weight,
	cl_mem* bias,
	const size_t in_width,
	const size_t out_width,
	const size_t resolution)
{
	cl_int err_num;
	cl_uint work_dim = 3;
	const size_t global_work_size[3] = { resolution, resolution, out_width };
	const size_t local_work_size[3] = { resolution, resolution, 1 };

	err_num = clSetKernelArg(kernel, 0, sizeof(cl_mem), input);
	err_num |= clSetKernelArg(kernel, 1, sizeof(cl_mem), output);
	err_num |= clSetKernelArg(kernel, 2, sizeof(cl_mem), weight);
	err_num |= clSetKernelArg(kernel, 3, sizeof(cl_mem), bias);
	err_num |= clSetKernelArg(kernel, 4, sizeof(unsigned short), &(unsigned short)in_width);
	err_num |= clSetKernelArg(kernel, 5, sizeof(unsigned short), &(unsigned short)out_width);
	err_num |= clSetKernelArg(kernel, 6, sizeof(unsigned char), &(unsigned char)resolution);
	err_num |= clSetKernelArg(kernel, 7, sizeof(float) * resolution * resolution, NULL);
	CHECK_ERROR(err_num);

	err_num = clEnqueueNDRangeKernel(command_queue, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	CHECK_ERROR(err_num);

	return;
}

static inline void maxp(
	cl_command_queue command_queue,
	cl_kernel kernel,
	cl_mem* input,
	cl_mem* output,
	const size_t width,
	const size_t out_resolution)
{
	cl_int err_num;
	cl_uint work_dim = 3;
	const size_t global_work_size[3] = { out_resolution, out_resolution, width };
	const size_t local_work_size[3] = { out_resolution, out_resolution, 1 };

	err_num = clSetKernelArg(kernel, 0, sizeof(cl_mem), input);
	err_num |= clSetKernelArg(kernel, 1, sizeof(cl_mem), output);
	err_num |= clSetKernelArg(kernel, 2, sizeof(unsigned short), &(unsigned short)width);
	err_num |= clSetKernelArg(kernel, 3, sizeof(unsigned char), &(unsigned char)out_resolution);
	CHECK_ERROR(err_num);

	err_num = clEnqueueNDRangeKernel(command_queue, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	CHECK_ERROR(err_num);

	return;
}

static inline void fc(
	cl_command_queue command_queue,
	cl_kernel kernel,
	cl_mem* input,
	cl_mem* output,
	cl_mem* weight,
	cl_mem* bias,
	const size_t num_innodes,
	const size_t num_outnodes)
{
	cl_int err_num;
	cl_uint work_dim = 2;
	const size_t global_work_size[2] = { num_innodes, num_outnodes };
	const size_t local_work_size[2] = { num_innodes, 1 };

	err_num = clSetKernelArg(kernel, 0, sizeof(cl_mem), input);
	err_num |= clSetKernelArg(kernel, 1, sizeof(cl_mem), output);
	err_num |= clSetKernelArg(kernel, 2, sizeof(cl_mem), weight);
	err_num |= clSetKernelArg(kernel, 3, sizeof(cl_mem), bias);
	err_num |= clSetKernelArg(kernel, 4, sizeof(unsigned short), &(unsigned short)num_innodes);
	err_num |= clSetKernelArg(kernel, 5, sizeof(unsigned short), &(unsigned short)num_outnodes);
	err_num |= clSetKernelArg(kernel, 6, sizeof(float) * num_innodes, NULL);
	CHECK_ERROR(err_num);

	err_num = clEnqueueNDRangeKernel(command_queue, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	CHECK_ERROR(err_num);

	return;
}

static void softmax(float arr[], const size_t size) {
	float max = arr[0];
	for (unsigned int i = 1; i < size; ++i)
		if (max < arr[i])
			max = arr[i];

	float sum = 0.0;
	for (unsigned int i = 0; i < size; ++i) {
		arr[i] = (float)exp(arr[i] - max);
		sum += arr[i];
	}

	for (unsigned int i = 0; i < size; ++i)
		arr[i] /= sum + (float)1e-7;
}

static int argmax(float arr[], size_t size) {
	unsigned int index = 0;
	float max = arr[index];
	for (unsigned int i = 1; i < size; ++i)
		if (max < arr[i]) {
			max = arr[i];
			index = i;
		}

	return (int)index;
}

static void initCL(
	cl_platform_id* platform,
	cl_uint* num_devices,
	cl_device_id* devices,
	cl_context* context,
	cl_command_queue* command_queue,
	cl_program* program)
{
	cl_int err_num;

	// Select device from user
	devices = selectPlatformAndDevices(platform, num_devices);

	// Note: Only one device will be used at this moment.
	if (*num_devices != 1) {
		printf("Invalid number of devices");
		exit(1);
	}

	// Create an OpenCL context
	printf("Creating a context...\n");
	*context = clCreateContext(NULL, *num_devices, devices, NULL, NULL, &err_num);
	CHECK_ERROR(err_num);

	// Create a command queue
	printf("Creating a command queue...\n");
	*command_queue = clCreateCommandQueue(*context, *devices, CL_QUEUE_PROFILING_ENABLE, &err_num);
	CHECK_ERROR(err_num);

	// Set optimal work group size
	setOptimalWorkGroupSize(*devices);

	// Create and build a program
	char build_option[128];
	sprintf(build_option, "-cl-fast-relaxed-math -D WPT=%zu", work_per_thread);
	*program = buildCLProgram(*context, (cl_uint)num_kernel_files, kernel_files, *num_devices, devices, build_option);
}

result* parallel(const images* images, const model* network) {
	cl_int err_num;

	result* output = loadResult(images->count, false);

	// Print device information
	printDeviceInfo();

	// Initialize OpenCL project
	cl_platform_id platform;
	cl_uint num_devices;
	cl_device_id devices;
	cl_context context;
	cl_command_queue command_queue;
	cl_program program;
	initCL(&platform, &num_devices, &devices, &context, &command_queue, &program);

	// Create kernel
	printf("Creating kernel \'%s\'...\n", "maxp");
	cl_kernel kernel_maxp = clCreateKernel(program, "maxp", &err_num);
	CHECK_ERROR(err_num);

	printf("Creating kernel \'%s\'...\n", "conv_low");
	cl_kernel kernel_conv_low = clCreateKernel(program, "conv_low", &err_num);
	CHECK_ERROR(err_num);

	printf("Creating kernel \'%s\'...\n", "conv_high");
	cl_kernel kernel_conv_high = clCreateKernel(program, "conv_high", &err_num);
	CHECK_ERROR(err_num);

	printf("Creating kernel \'%s\'...\n", "fc");
	cl_kernel kernel_fc = clCreateKernel(program, "fc", &err_num);
	CHECK_ERROR(err_num);

	// Allocate memory for feature maps (sequential code)
	printf("Allocating temporary memory for feature maps... (Host)\n");
	float* fmaps[21];
	for (int i = 0; i < 21; ++i)
		fmaps[i] = (float*)malloc_c(sizeof(float) * RES[i] * RES[i] * WIDTHS[i][1]);

	// Declare memory objects for (OpenCL)
	// input images
	cl_mem* mem_images = (cl_mem*)malloc_c(sizeof(cl_mem) * images->count);
	// feature maps
	cl_mem* mem_fmaps = (cl_mem*)malloc_c(sizeof(cl_mem) * 21);
	// weights and biases
	cl_mem* mem_weights = (cl_mem*)malloc_c(sizeof(cl_mem) * 21);
	cl_mem* mem_biases = (cl_mem*)malloc_c(sizeof(cl_mem) * 21);
	// output of last fc layer
	cl_mem* mem_results = (cl_mem*)malloc_c(sizeof(cl_mem) * images->count);

	printf("Operation started on the OpenCL device. Please wait...\n");

	// Stopwatch starts here.
	output->start_time = clock();

	// Create memory objects and write data to memory them
	for (int i = 0; i < 21; ++i) {
		// Create memory objects for feature maps
		size_t fmap_size = sizeof(float) * RES[i] * RES[i] * WIDTHS[i][1];
		mem_fmaps[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, fmap_size, NULL, &err_num);
		CHECK_ERROR(err_num);

		// Create memory objects for weights and biases
		// and write paramter values to them
		if (i < 18) {
			if (i == 2 || i == 5 || i == 9 || i == 13 || i == 17) {
				mem_weights[i] = NULL;
				mem_biases[i] = NULL;
			}
			else {
				size_t weight_size = sizeof(float) * 9 * WIDTHS[i][0] * WIDTHS[i][1];
				mem_weights[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, weight_size, NULL, &err_num);
				err_num |= clEnqueueWriteBuffer(command_queue, mem_weights[i], CL_FALSE, 0, weight_size, network->weights[i], 0, NULL, NULL);
				CHECK_ERROR(err_num);

				size_t bias_size = sizeof(float) * WIDTHS[i][1];
				mem_biases[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, bias_size, NULL, &err_num);
				err_num |= clEnqueueWriteBuffer(command_queue, mem_biases[i], CL_FALSE, 0, bias_size, network->biases[i], 0, NULL, NULL);
				CHECK_ERROR(err_num);
			}
		}
		else {
			size_t weight_size = sizeof(float) * WIDTHS[i][0] * WIDTHS[i][1];
			mem_weights[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, weight_size, NULL, &err_num);
			err_num |= clEnqueueWriteBuffer(command_queue, mem_weights[i], CL_FALSE, 0, weight_size, network->weights[i], 0, NULL, NULL);
			CHECK_ERROR(err_num);

			size_t bias_size = sizeof(float) * WIDTHS[i][1];
			mem_biases[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, bias_size, NULL, &err_num);
			err_num |= clEnqueueWriteBuffer(command_queue, mem_biases[i], CL_FALSE, 0, bias_size, network->biases[i], 0, NULL, NULL);
			CHECK_ERROR(err_num);
		}
	}
	
	const size_t input_size = sizeof(float) * 32 * 32 * 3;
	const size_t output_size = sizeof(float) * 10;
	for (unsigned int i = 0; i < images->count; ++i) {
		mem_images[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, input_size, NULL, &err_num);
		err_num |= clEnqueueWriteBuffer(command_queue, mem_images[i], CL_FALSE, 0, input_size, images->at[i], 0, NULL, NULL);
		CHECK_ERROR(err_num);

		float* buffer = images->at[i];
		cl_mem mem_buffer = mem_images[i];
		for (unsigned int layer = 0; layer < 21; ++layer) {
			const float* weight = network->weights[layer];
			const float* bias = network->biases[layer];
			const size_t in_width = WIDTHS[layer][0];
			const size_t out_width = WIDTHS[layer][1];
			const size_t res = RES[layer];

			if (layer < 18) {
				if (layer == 2 || layer == 5 || layer == 9 || layer == 13 || layer == 17) {
					maxp(command_queue, kernel_maxp, &mem_buffer, &mem_fmaps[layer], out_width, res);
				}
				else {
					if (layer < 9)
						convolutionLow(command_queue, kernel_conv_low, &mem_buffer, &mem_fmaps[layer], &mem_weights[layer], &mem_biases[layer], in_width, out_width, res);
					else
						convolutionHigh(command_queue, kernel_conv_high, &mem_buffer, &mem_fmaps[layer], &mem_weights[layer], &mem_biases[layer], in_width, out_width, res);
				}
			}
			else {
				fc(command_queue, kernel_fc, &mem_buffer, &mem_fmaps[layer], &mem_weights[layer], &mem_biases[layer], in_width, out_width);
			}

			buffer = fmaps[layer];
			mem_buffer = mem_fmaps[layer];
		}

		err_num = clEnqueueReadBuffer(command_queue, mem_buffer, CL_TRUE, 0, output_size, buffer, 0, NULL, NULL);
		CHECK_ERROR(err_num);

		softmax(buffer, 10);

		output->labels[i] = argmax(buffer, 10);
		output->confs[i] = buffer[output->labels[i]];
	}
	// Stopwatch stops here.
	output->end_time = clock();
	printf("Done.        \n");

	for (int i = 0; i < 21; ++i)
		free_c(fmaps[i]);

	return output;
}
