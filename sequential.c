#include <stdio.h>
#include <string.h>
#include <math.h>
#include "cnn.h"
#include "customlib.h"

#define ReLU(x) (x) > 0 ? (x) : 0

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

		for (unsigned int act = 0; act < channel_size; ++act) {
			*output = ReLU(*output + *bias);
			++output;
		}
		++bias;
	}
}

static void maxp(const float input[], float output[], const size_t width, const size_t out_resolution) {
	const size_t in_resolution = out_resolution * 2;
	const size_t in_channel_size = in_resolution * in_resolution;

	const float* in_channel = input;
	for (unsigned int ch = 0; ch < width; ++ch) {
		for (unsigned int out_y = 0; out_y < out_resolution; ++out_y) {
			for (unsigned int out_x = 0; out_x < out_resolution; ++out_x) {
				float max = 0.0;
				for (unsigned int dy = 0; dy < 2; ++dy) {
					for (unsigned int dx = 0; dx < 2; ++dx) {
						int in_y = 2 * out_y + dy;
						int in_x = 2 * out_x + dx;
						float act = in_channel[in_y * in_resolution + in_x];
						if (max < act)
							max = act;
					}
				}
				*output++ = max;
			}
		}
		in_channel += in_channel_size;
	}
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

		*write_to += *bias++;

		*write_to = ReLU(*write_to);

		++write_to;
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

void sequential(const images* images, const model* network, int labels[], float confidences[]) {
	float* fmaps[21];
	for (int i = 0; i < 21; ++i)
		fmaps[i] = (float*)malloc_c(sizeof(float) * RES[i] * RES[i] * WIDTHS[i][1]);

	for (unsigned int i = 0; i < images->count; ++i) {
		printf("Running image inference on CPU... [%u/%zd]\t", i + 1, images->count);
		const float* buffer = images->at[i];
		for (unsigned int layer = 0; layer < 21; ++layer) {
			const float* weight = network->weights[layer];
			const float* bias = network->biases[layer];
			const size_t in_width = WIDTHS[layer][0];
			const size_t out_width = WIDTHS[layer][1];
			const size_t res = RES[layer];

			if (layer < 18) {
				if (layer == 2 || layer == 5 || layer == 9 || layer == 13 || layer == 17)
					maxp(buffer, fmaps[layer], out_width, res);
				else
					convolution(buffer, fmaps[layer], weight, bias, in_width, out_width, res);
			}
			else
				fc(buffer, fmaps[layer], weight, bias, in_width, out_width);

			buffer = fmaps[layer];
		}

		softmax(fmaps[20], 10);

		labels[i] = argmax(fmaps[20], 10);
		confidences[i] = fmaps[20][labels[i]];
		printf("\r");
	}
	printf("\nDone.\n");

	for (int i = 0; i < 21; ++i)
		free_c(fmaps[i]);

	return;
}
