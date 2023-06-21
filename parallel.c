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
	cl_mem mem_label = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int), NULL, &err_num);
	CHECK_ERROR(err_num);
	cl_mem mem_conf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float), NULL, &err_num);
	CHECK_ERROR(err_num);

	// Create memory objects and write data to memory them
	printf("Loading the model...\n");
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
	
	printf("Operation started on the OpenCL device. Please wait.");

	// Stopwatch starts here.
	output->start_time = clock();
	printf(".");

	const size_t input_size = sizeof(float) * NUM_PIXELS;
	const size_t output_size = sizeof(float) * NUM_CLASSES;
	for (unsigned int i = 0; i < images->count; ++i) {
		mem_images[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, input_size, NULL, &err_num);
		err_num |= clEnqueueWriteBuffer(command_queue, mem_images[i], CL_FALSE, 0, input_size, images->at[i], 0, NULL, NULL);
		CHECK_ERROR(err_num);

		convolutionLow(command_queue, kernel_conv_low, &mem_images[i], &mem_fmaps[0], &mem_weights[0], &mem_biases[0], WIDTHS[0][0], WIDTHS[0][1], RES[0]);
		convolutionLow(command_queue, kernel_conv_low, &mem_fmaps[0], &mem_fmaps[1], &mem_weights[1], &mem_biases[1], WIDTHS[1][0], WIDTHS[1][1], RES[1]);
		maxp(command_queue, kernel_maxp, &mem_fmaps[1], &mem_fmaps[2], WIDTHS[2][1], RES[2]);

		convolutionLow(command_queue, kernel_conv_low, &mem_fmaps[2], &mem_fmaps[3], &mem_weights[3], &mem_biases[3], WIDTHS[3][0], WIDTHS[3][1], RES[3]);
		convolutionLow(command_queue, kernel_conv_low, &mem_fmaps[3], &mem_fmaps[4], &mem_weights[4], &mem_biases[4], WIDTHS[4][0], WIDTHS[4][1], RES[4]);
		maxp(command_queue, kernel_maxp, &mem_fmaps[4], &mem_fmaps[5], WIDTHS[5][1], RES[5]);

		convolutionLow(command_queue, kernel_conv_low, &mem_fmaps[5], &mem_fmaps[6], &mem_weights[6], &mem_biases[6], WIDTHS[6][0], WIDTHS[6][1], RES[6]);
		convolutionLow(command_queue, kernel_conv_low, &mem_fmaps[6], &mem_fmaps[7], &mem_weights[7], &mem_biases[7], WIDTHS[7][0], WIDTHS[7][1], RES[7]);
		convolutionLow(command_queue, kernel_conv_low, &mem_fmaps[7], &mem_fmaps[8], &mem_weights[8], &mem_biases[8], WIDTHS[8][0], WIDTHS[8][1], RES[8]);
		maxp(command_queue, kernel_maxp, &mem_fmaps[8], &mem_fmaps[9], WIDTHS[9][1], RES[9]);

		convolutionHigh(command_queue, kernel_conv_high, &mem_fmaps[9], &mem_fmaps[10], &mem_weights[10], &mem_biases[10], WIDTHS[10][0], WIDTHS[10][1], RES[10]);
		convolutionHigh(command_queue, kernel_conv_high, &mem_fmaps[10], &mem_fmaps[11], &mem_weights[11], &mem_biases[11], WIDTHS[11][0], WIDTHS[11][1], RES[11]);
		convolutionHigh(command_queue, kernel_conv_high, &mem_fmaps[11], &mem_fmaps[12], &mem_weights[12], &mem_biases[12], WIDTHS[12][0], WIDTHS[12][1], RES[12]);
		maxp(command_queue, kernel_maxp, &mem_fmaps[12], &mem_fmaps[13], WIDTHS[13][1], RES[13]);

		convolutionHigh(command_queue, kernel_conv_high, &mem_fmaps[13], &mem_fmaps[14], &mem_weights[14], &mem_biases[14], WIDTHS[14][0], WIDTHS[14][1], RES[14]);
		convolutionHigh(command_queue, kernel_conv_high, &mem_fmaps[14], &mem_fmaps[15], &mem_weights[15], &mem_biases[15], WIDTHS[15][0], WIDTHS[15][1], RES[15]);
		convolutionHigh(command_queue, kernel_conv_high, &mem_fmaps[15], &mem_fmaps[16], &mem_weights[16], &mem_biases[16], WIDTHS[16][0], WIDTHS[16][1], RES[16]);
		maxp(command_queue, kernel_maxp, &mem_fmaps[16], &mem_fmaps[17], WIDTHS[17][1], RES[17]);

		fc(command_queue, kernel_fc, &mem_fmaps[17], &mem_fmaps[18], &mem_weights[18], &mem_biases[18], WIDTHS[18][0], WIDTHS[18][1]);
		fc(command_queue, kernel_fc, &mem_fmaps[18], &mem_fmaps[19], &mem_weights[19], &mem_biases[19], WIDTHS[19][0], WIDTHS[19][1]);
		fc(command_queue, kernel_fc, &mem_fmaps[19], &mem_fmaps[20], &mem_weights[20], &mem_biases[20], WIDTHS[20][0], WIDTHS[20][1]);

		argmax(command_queue, kernel_argmax, &mem_fmaps[20], &mem_label, &mem_conf, NUM_CLASSES);

		err_num = clEnqueueReadBuffer(command_queue, mem_label, CL_FALSE, 0, sizeof(int), &output->labels[i], 0, NULL, NULL);
		err_num |= clEnqueueReadBuffer(command_queue, mem_conf, CL_FALSE, 0, sizeof(float), &output->confs[i], 0, NULL, NULL);
		CHECK_ERROR(err_num);
	}
	printf(".");

	err_num = clFinish(command_queue);
	CHECK_ERROR(err_num);
	printf("\n");

	// Stopwatch stops here.
	output->end_time = clock();
	printf("Done.        \n");

	// Release memory objects
	err_num = clReleaseMemObject(mem_label);
	err_num |= clReleaseMemObject(mem_conf);
	for (int i = 0; i < 21; ++i) {
		err_num |= clReleaseMemObject(mem_fmaps[i]);

		if (i == 2 || i == 5 || i == 9 || i == 13 || i == 17) continue;

		err_num |= clReleaseMemObject(mem_weights[i]);
		err_num |= clReleaseMemObject(mem_biases[i]);
	}
	CHECK_ERROR(err_num);

	// De-allocate memories
	free_c(mem_images);
	free_c(mem_fmaps);
	free_c(mem_weights);
	free_c(mem_biases);

	// Release kernels
	err_num = clReleaseKernel(kernel_maxp);
	err_num |= clReleaseKernel(kernel_conv_low);
	err_num |= clReleaseKernel(kernel_conv_high);
	err_num |= clReleaseKernel(kernel_fc);
	err_num |= clReleaseKernel(kernel_argmax);
	CHECK_ERROR(err_num);

	// Release the context, command_queue, and the program
	termCL(&context, &command_queue, &program);

	return output;
}
