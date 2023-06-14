#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>
#include "cnn.h"
#include "customlib.h"
#include "clproject.h"

#define BATCH 1
#define NUM_PIXELS (32 * 32 * 3)
#define NUM_CLASSES 10

static size_t max_work_group_size;
static size_t work_per_thread;
static const size_t num_kernel_files = 4;

const char* kernel_files[] = {
	"conv.cl",
	"maxp.cl",
	"fc.cl",
	"argmax.cl"
};

static inline float ReLU(float val) {
	if (val > 0.0)	return	val;
	else			return	0.0;
}

static void setWPT(cl_device_id device) {
	cl_int err_num;

	err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
	CHECK_ERROR(err_num);

	if (max_work_group_size >= 1024)
		work_per_thread = 1;
	else if (max_work_group_size >= 256)
		work_per_thread = 2;
	else
		work_per_thread = 4;
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
	const size_t global_work_size[2] = { num_innodes / work_per_thread, num_outnodes };
	const size_t local_work_size[2] = { num_innodes / work_per_thread, 1 };

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

// Softmax is internally calculated in argmax kernel
static inline void argmax(
	cl_command_queue command_queue,
	cl_kernel kernel,
	cl_mem* input,
	cl_mem* label,
	cl_mem* confidence,
	const size_t input_size)
{
	cl_int err_num;
	cl_uint work_dim = 1;
	const size_t global_work_size[1] = { 1 };
	const size_t local_work_size[1] = { 1 };

	err_num = clSetKernelArg(kernel, 0, sizeof(cl_mem), input);
	err_num |= clSetKernelArg(kernel, 1, sizeof(cl_mem), label);
	err_num |= clSetKernelArg(kernel, 2, sizeof(cl_mem), confidence);
	CHECK_ERROR(err_num);

	err_num = clEnqueueNDRangeKernel(command_queue, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	CHECK_ERROR(err_num);

	return;
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
	setWPT(*devices);

	// Create and build a program
	char build_option[128];
	sprintf(build_option, "-cl-fast-relaxed-math -D WPT=%zu -D NUM_CLASSES=%d", work_per_thread, NUM_CLASSES);
	*program = buildCLProgram(*context, (cl_uint)num_kernel_files, kernel_files, *num_devices, devices, build_option);
}

static void termCL(
	cl_context* context,
	cl_command_queue* command_queue,
	cl_program* program)
{
	cl_int err_num;

	err_num = clReleaseProgram(*program);
	err_num |= clReleaseCommandQueue(*command_queue);
	err_num |= clReleaseContext(*context);
	CHECK_ERROR(err_num);

	return;
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

	printf("Creating kernel \'%s\'...\n", "argmax");
	cl_kernel kernel_argmax = clCreateKernel(program, "argmax", &err_num);
	CHECK_ERROR(err_num);

	// Declare memory objects for (OpenCL)
	// input images
	cl_mem* mem_images = (cl_mem*)malloc_c(sizeof(cl_mem) * images->count);
	// feature maps
	cl_mem* mem_fmaps = (cl_mem*)malloc_c(sizeof(cl_mem) * 21);
	// weights and biases
	cl_mem* mem_weights = (cl_mem*)malloc_c(sizeof(cl_mem) * 21);
	cl_mem* mem_biases = (cl_mem*)malloc_c(sizeof(cl_mem) * 21);
	// output from argmax
	cl_mem* mem_labels = (cl_mem*)malloc_c(sizeof(cl_mem) * images->count);
	cl_mem* mem_confs = (cl_mem*)malloc_c(sizeof(cl_mem) * images->count);

	// Create OpenCL event to wait for buffer reading
	cl_event* event_rlabels = (cl_event*)malloc_c(sizeof(cl_event) * images->count);
	cl_event* event_rconfs = (cl_event*)malloc_c(sizeof(cl_event) * images->count);

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
	
	const size_t input_size = sizeof(float) * NUM_PIXELS;
	const size_t output_size = sizeof(float) * NUM_CLASSES;
	for (unsigned int i = 0; i < images->count; ++i) {
		mem_images[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, input_size, NULL, &err_num);
		err_num |= clEnqueueWriteBuffer(command_queue, mem_images[i], CL_FALSE, 0, input_size, images->at[i], 0, NULL, NULL);
		CHECK_ERROR(err_num);

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

			mem_buffer = mem_fmaps[layer];
		}

		mem_labels[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int), NULL, &err_num);
		CHECK_ERROR(err_num);

		mem_confs[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float), NULL, &err_num);
		CHECK_ERROR(err_num);

		argmax(command_queue, kernel_argmax, &mem_buffer, &mem_labels[i], &mem_confs[i], NUM_CLASSES);

		err_num = clEnqueueReadBuffer(command_queue, mem_labels[i], CL_FALSE, 0, sizeof(int), &output->labels[i], 0, NULL, &event_rlabels[i]);
		err_num |= clEnqueueReadBuffer(command_queue, mem_confs[i], CL_FALSE, 0, sizeof(float), &output->confs[i], 0, NULL, &event_rconfs[i]);
		CHECK_ERROR(err_num);
	}

	err_num = clEnqueueWaitForEvents(command_queue, (cl_uint)images->count, event_rlabels);
	err_num |= clEnqueueWaitForEvents(command_queue, (cl_uint)images->count, event_rconfs);
	CHECK_ERROR(err_num);

	// Stopwatch stops here.
	output->end_time = clock();
	printf("Done.        \n");

	for (int i = 0; i < 21; ++i) {
		err_num |= clReleaseMemObject(mem_fmaps[i]);

		if (i == 2 || i == 5 || i == 9 || i == 13 || i == 17) continue;

		err_num |= clReleaseMemObject(mem_weights[i]);
		err_num |= clReleaseMemObject(mem_biases[i]);
	}
	CHECK_ERROR(err_num);

	for (int i = 0; i < images->count; ++i) {
		err_num |= clReleaseEvent(event_rlabels[i]);
		err_num |= clReleaseEvent(event_rconfs[i]);
	}
	CHECK_ERROR(err_num);

	free_c(mem_images);
	free_c(mem_fmaps);
	free_c(mem_weights);
	free_c(mem_biases);
	free_c(mem_labels);
	free_c(mem_confs);
	free_c(event_rlabels);
	free_c(event_rconfs);

	err_num = clReleaseKernel(kernel_maxp);
	err_num |= clReleaseKernel(kernel_conv_low);
	err_num |= clReleaseKernel(kernel_conv_high);
	err_num |= clReleaseKernel(kernel_fc);
	err_num |= clReleaseKernel(kernel_argmax);
	CHECK_ERROR(err_num);

	termCL(&context, &command_queue, &program);

	return output;
}
