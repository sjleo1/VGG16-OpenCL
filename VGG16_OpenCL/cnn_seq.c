#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <windows.h>
#include <math.h>
#include <time.h>
#include <direct.h>
#include "cnn.h"

extern const char* CLASS_NAME[];
extern const int INPUT_DIM[];
extern const int OUTPUT_DIM[];
extern const int NBYN[];

static void convolution(float* inputs, float* outputs, float* filters, float* biases, int indim, int outdim, int nbyn) {
	memset(outputs, 0, nbyn * nbyn * outdim * sizeof(float));

	float* output_ptr = outputs;
	int input_offset = nbyn * nbyn;

	for (int outneuron = 0; outneuron < outdim; outneuron++) {
		float* input = inputs;
		for (int inneuron = 0; inneuron < indim; inneuron++) {
			float* output = outputs;
			for (int row = 0; row < nbyn; row++) {
				for (int col = 0; col < nbyn; col++) {
					float sum = 0.0f;
					for (int frow = 0; frow < 3; frow++) {
						int y = row + frow - 1;
						for (int fcol = 0; fcol < 3; fcol++) {
							int x = col + fcol - 1;
							if (x >= 0 && x < nbyn && y >= 0 && y < nbyn)
								sum += input[y * nbyn + x] * filters[frow * 3 + fcol];
						}
					}
					*(output++) += sum;
				}
			}
			filters += 9;
			input += input_offset;
		}
		for (int i = 0; i < input_offset; i++, outputs++) {
			*outputs += *biases;
			if (*outputs < 0)
				*outputs = 0;
		}
		biases++;
	}

	return;
}

static void maxPooling(float* input, float* output, int dim, int nbyn) {
	while (dim--) {
		for (int row = 0; row < nbyn; row += 2) {
			for (int col = 0; col < nbyn; col += 2) {
				float max = 0.0f;
				for (int y = 0; y < 2; y++) {
					for (int x = 0; x < 2; x++) {
						float temp = input[(row + y) * nbyn + col + x];
						if (max < temp)
							max = temp;
					}
				}
				*(output++) = max;
			}
		}
		input += nbyn * nbyn;
	}

	return;
}

static void FCLayer(float* inputs, float* outputs, float* weights, float* biases, int indim, int outdim) {
	for (int outneuron = 0; outneuron < outdim; outneuron++) {
		float sum = 0.0f;
		for (int inneuron = 0; inneuron < indim; inneuron++)
			sum += inputs[inneuron] * (*weights++);
		sum += biases[outneuron];

		if (sum > 0.0f)
			outputs[outneuron] = sum;
		else
			outputs[outneuron] = 0.0f;
	}

	return;
}

static void softmax(float* inputs, int N) {
	float max = *inputs;
	for (int i = 1; i < N; i++)
		if (max < inputs[i])
			max = inputs[i];

	float sum = 0.0f;
	for (int i = 0; i < N; i++)
		sum += exp(inputs[i] - max);

	for (int i = 0; i < N; i++)
		inputs[i] = exp(inputs[i] - max) / (sum + 1e-7);

	return;
}

static int findMax(float* inputs, int class_num) {
	int max = 0, maxindex;

	for (int i = 0; i < class_num; i++)
		if (max < inputs[i]) {
			max = inputs[i];
			maxindex = i;
		}

	return maxindex;
}

void sequentialCNN(float* images, float* networks, int* labels, float* confidences, int num_of_image) {
	float* filters[21], * biases[21];
	int offset = 0;

	// Setting wieghts and biases for convolution layers
	for (int i = 0; i < 17; i++) {
		if (i == 2 || i == 5 || i == 9 || i == 13)
			continue;

		filters[i] = networks + offset;
		offset += 3 * 3 * INPUT_DIM[i] * OUTPUT_DIM[i];

		biases[i] = networks + offset;
		offset += OUTPUT_DIM[i];
	}
	// Setting weights and biases for fully connected layers
	for (int i = 18; i < 21; i++) {
		filters[i] = networks + offset;
		offset += INPUT_DIM[i] * OUTPUT_DIM[i];

		biases[i] = networks + offset;
		offset += OUTPUT_DIM[i];
	}

	float* layer[21];
	for (int i = 0; i < 21; i++) {
		layer[i] = (float*)malloc(sizeof(float) * OUTPUT_DIM[i] * NBYN[i] * NBYN[i]);
		if (layer[i] == NULL) {
			perror("malloc() error.\n");
			exit(1);
		}
	}

	time_t start;
	start = clock();

	for (int i = 0; i < num_of_image; i++) {
		printf("\r%d/%d", i, num_of_image);

		convolution(images, layer[0], filters[0], biases[0], INPUT_DIM[0], OUTPUT_DIM[0], NBYN[0]);
		convolution(layer[0], layer[1], filters[1], biases[1], INPUT_DIM[1], OUTPUT_DIM[1], NBYN[1]);
		maxPooling(layer[1], layer[2], INPUT_DIM[2], NBYN[2] * 2);

		convolution(layer[2], layer[3], filters[3], biases[3], INPUT_DIM[3], OUTPUT_DIM[3], NBYN[3]);
		convolution(layer[3], layer[4], filters[4], biases[4], INPUT_DIM[4], OUTPUT_DIM[4], NBYN[4]);
		maxPooling(layer[4], layer[5], INPUT_DIM[5], NBYN[5] * 2);

		convolution(layer[5], layer[6], filters[6], biases[6], INPUT_DIM[6], OUTPUT_DIM[6], NBYN[6]);
		convolution(layer[6], layer[7], filters[7], biases[7], INPUT_DIM[7], OUTPUT_DIM[7], NBYN[7]);
		convolution(layer[7], layer[8], filters[8], biases[8], INPUT_DIM[8], OUTPUT_DIM[8], NBYN[8]);
		maxPooling(layer[8], layer[9], INPUT_DIM[9], NBYN[9] * 2);

		convolution(layer[9], layer[10], filters[10], biases[10], INPUT_DIM[10], OUTPUT_DIM[10], NBYN[10]);
		convolution(layer[10], layer[11], filters[11], biases[11], INPUT_DIM[11], OUTPUT_DIM[11], NBYN[11]);
		convolution(layer[11], layer[12], filters[12], biases[12], INPUT_DIM[12], OUTPUT_DIM[12], NBYN[12]);
		maxPooling(layer[12], layer[13], INPUT_DIM[13], NBYN[13] * 2);

		convolution(layer[13], layer[14], filters[14], biases[14], INPUT_DIM[14], OUTPUT_DIM[14], NBYN[14]);
		convolution(layer[14], layer[15], filters[15], biases[15], INPUT_DIM[15], OUTPUT_DIM[15], NBYN[15]);
		convolution(layer[15], layer[16], filters[16], biases[16], INPUT_DIM[16], OUTPUT_DIM[16], NBYN[16]);
		maxPooling(layer[16], layer[17], INPUT_DIM[17], NBYN[17] * 2);

		fc_layer(layer[17], layer[18], filters[18], biases[18], INPUT_DIM[18], OUTPUT_DIM[18]);
		fc_layer(layer[18], layer[19], filters[19], biases[19], INPUT_DIM[19], OUTPUT_DIM[19]);
		fc_layer(layer[19], layer[20], filters[20], biases[20], INPUT_DIM[20], OUTPUT_DIM[20]);

		softmax(layer[20], 10);

		labels[i] = findMax(layer[20], 10);
		confidences[i] = layer[20][labels[i]];
		images += 32 * 32 * 3;
	}

	time_t end = clock();

	printf("Elapsed time: %.2f sec.\n", (double)(end - start) / CLK_TCK);

	for (int i = 0; i < 21; i++)
		free(layer[i]);

	return;
}