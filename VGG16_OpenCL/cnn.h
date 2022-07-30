#pragma once
#pragma warning (disable: 4996 6031)
#define CHECK_ERROR(err)\
	if (err != CL_SUCCESS) {\
		printf("[%s:%d] OpenCL error %d.\n", __FILE__, __LINE__, err);\
		exit(EXIT_FAILURE);\
	}

void sequentialCNN(float* images, float* network, int* labels, float* confidences, int num_of_image);
void parallelCNNInit();
void parallelCNN(float* images, float** network, int* labels, float* confidences, int num_images);