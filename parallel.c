#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>
#include "cnn.h"
#include "customlib.h"
#include "clproject.h"

const size_t num_kernel_files = 1;
const char* kernel_files[] = {
	"maxp.cl"
};

char build_option[128] = "-cl-fast-relaxed-math";

static inline float ReLU(float val) {
	if (val > 0.0)	return	val;
	else			return	0.0;
}

static void convolution(
	const float input[], float output[],
	const float weight[], const float bias[],
	const size_t in_width, const size_t out_width, const size_t resolution)
{
	const size_t channel_size = resolution * resolution;

	memset(output, 0, sizeof(float) * channel_size * out_width);
	const float* filter = weight;
	for (unsigned int out_c = 0; out_c < out_width; ++out_c) {
		const float* in_channel = input;
		for (unsigned int in_c = 0; in_c < in_width; ++in_c) {
			float* write_to = output;
			for (unsigned int out_y = 0; out_y < resolution; ++out_y) {
				for (int out_x = 0; out_x < resolution; ++out_x) {
					float buffer = 0.0;
					for (unsigned int f_y = 0; f_y < 3; ++f_y) {
						for (unsigned int f_x = 0; f_x < 3; ++f_x) {
							int in_y = out_y + f_y - 1;
							int in_x = out_x + f_x - 1;
							if (in_y >= 0 && in_y < resolution && in_x >= 0 && in_x < resolution) {
								buffer += in_channel[in_y * resolution + in_x] * filter[f_y * 3 + f_x];
							}
						}
					}
					*write_to++ += buffer;
				}
			}

			filter += 9;
			in_channel += channel_size;
		}

		for (unsigned int act = 0; act < channel_size; ++act)
			*output++ = ReLU(*output + *bias);
		++bias;
	}
}

static inline void maxp(cl_command_queue command_queue, cl_kernel kernel, cl_mem* input, cl_mem* output, const size_t width, const size_t out_resolution) {
	cl_int err_num;
	cl_uint work_dim = 3;
	const size_t global_work_size[3] = { out_resolution, out_resolution, width };
	const size_t local_work_size[3] = { 1, 1, 1 };

	err_num = clSetKernelArg(kernel, 0, sizeof(cl_mem), input);
	err_num |= clSetKernelArg(kernel, 1, sizeof(cl_mem), output);
	err_num |= clSetKernelArg(kernel, 2, sizeof(unsigned short), &(unsigned short)width);
	err_num |= clSetKernelArg(kernel, 3, sizeof(unsigned char), &(unsigned char)out_resolution);
	CHECK_ERROR(err_num);

	err_num = clEnqueueNDRangeKernel(command_queue, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	CHECK_ERROR(err_num);

	return;
}

static void fc(
	const float input[], float output[],
	const float weight[], const float bias[],
	const size_t num_innodes, const size_t num_outnodes)
{
	memset(output, 0, sizeof(float) * num_outnodes);

	float* write_to = output;
	for (unsigned int out_n = 0; out_n < num_outnodes; ++out_n) {
		for (unsigned int in_n = 0; in_n < num_innodes; ++in_n)
			*write_to += input[in_n] * *weight++;
		*write_to++ = ReLU(*write_to + *bias++);
	}
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

	// Create and build a program
	*program = buildCLProgram(*context, (cl_uint)num_kernel_files, kernel_files, *num_devices, devices, build_option);
}

void parallel(const images* images, const model* network, int labels[], float confidences[]) {
	cl_int err_num;

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

	// Create memory objects and write data to memory them
	printf("Copying data to memory objects...\n");
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

	printf("Running image inference on the OpenCL devices...\n");
	for (unsigned int i = 0; i < images->count; ++i) {
		printf("[%u/%zd]", i + 1, images->count);
		const float* buffer = images->at[i];
		cl_mem mem_buffer = mem_images[i];
		for (unsigned int layer = 0; layer < 21; ++layer) {
			const float* weight = network->weights[layer];
			const float* bias = network->biases[layer];
			const size_t in_width = WIDTHS[layer][0];
			const size_t out_width = WIDTHS[layer][1];
			const size_t res = RES[layer];

			if (layer < 18) {
				if (layer == 2 || layer == 5 || layer == 9 || layer == 13 || layer == 17) {
					const size_t in_size = sizeof(float) * res * res * 4 * in_width;
					const size_t out_size = sizeof(float) * res * res * out_width;

					err_num = clEnqueueWriteBuffer(command_queue, mem_buffer, CL_FALSE, 0, in_size, buffer, 0, NULL, NULL);
					CHECK_ERROR(err_num);

					maxp(command_queue, kernel_maxp, &mem_buffer, &mem_fmaps[layer], out_width, res);

					err_num = clEnqueueReadBuffer(command_queue, mem_fmaps[layer], CL_TRUE, 0, out_size, fmaps[layer], 0, NULL, NULL);
					CHECK_ERROR(err_num);
				}
				else
					convolution(buffer, fmaps[layer], weight, bias, in_width, out_width, res);
			}
			else
				fc(buffer, fmaps[layer], weight, bias, in_width, out_width);

			buffer = fmaps[layer];
			mem_buffer = mem_fmaps[layer];
		}

		softmax(fmaps[20], 10);

		labels[i] = argmax(fmaps[20], 10);
		confidences[i] = fmaps[20][labels[i]];
		printf("\r");
	}
	printf("Done.        \n");

	for (int i = 0; i < 21; ++i)
		free_c(fmaps[i]);

	return;
}
