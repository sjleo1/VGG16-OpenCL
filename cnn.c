#include <stdio.h>
#include "cnn.h"
#include "customlib.h"

const size_t model_size = 60980520;

// `images.bin`: A binary file of the
// CIFAR 10 dataset
const char image_file[] = "images.bin";

// `labels.bin`: Labels of the 10000 images of
// the CIFAR 10 dataset
const char label_file[] = "labels.bin";

// `network.bin`: Pre-trained weights and biases
// of the model
const char network_file[] = "network.bin";

// `answer.bin`: Target result of the operation.
// It contains the predicted labels and confidences
// of the operation. Note that the result of the
// operation should match with the answer file, not
// the labels file.
const char answer_file[] = "answer.bin";

const char* CLASS_NAME[] = {
	"airplane",
	"automobile",
	"bird",
	"cat",
	"deer",
	"dog",
	"frog",
	"horse",
	"ship",
	"truck"
};

/*
* In this project, `WIDTH` and "width" represents
* the number of channels in each feature map, also
* called as "depth" depending on the situation.
*
* Not to be confused with the horizontal length
* of a feature map, which will be refered to as
* resolution or `RES` throughout this project.
*/
const size_t WIDTHS[][2] = {
	/*
	{input width, output width}
					   conv<receptive field size>-<# channels>
	*/
	{ 3, 64 },		// conv3-64 (1-1)
	{ 64, 64 },		// conv3-64 (1-2)
	{ 64, 64 },		// maxpool

	{ 64, 128 },	// conv3-128 (2-1)
	{ 128, 128 },	// conv3-128 (2-2)
	{ 128, 128 },	// maxpool

	{ 128, 256 },	// conv3-256 (3-1)
	{ 256, 256 },	// conv3-256 (3-2)
	{ 256, 256 },	// conv3-256 (3-3)
	{ 256, 256 },	// maxpool

	{ 256, 512 },	// conv3-512 (4-1)
	{ 512, 512 },	// conv3-512 (4-2)
	{ 512, 512 },	// conv3-512 (4-3)
	{ 512, 512 },	// maxpool

	{ 512, 512 },	// conv3-512 (5-1)
	{ 512, 512 },	// conv3-512 (5-2)
	{ 512, 512 },	// conv3-512 (5-3)
	{ 512, 512 },	// maxpool

	{ 512, 512 },	// fc-512
	{ 512, 512 },	// fc-512
	{ 512, 10 }		// fc-10
};

/*
* `RES` and resolution refer to the horizontal
* and vertical length of a feature map, which
* also can be called as "width" and "height".
*
* However, the term "width" in this project
* refers to the number of channels.
*
* All channels of input images and feature maps
* are square except for that in the fully
* connected layers.
*/
const size_t RES[] = {
	/* Note for maxpool: It's output resolution. */
	32, 32,		// conv
	16,			// maxpool

	16, 16,		// conv
	8,			// maxpool

	8, 8, 8,	// conv
	4,			// maxpool

	4, 4, 4,	// conv
	2,			// maxpool

	2, 2, 2,	// conv
	1,			// maxpool

	1, 1, 1		// fc
};

void* readByte(const char* file_name, size_t size) {
	FILE* stream = fopen_c(file_name, "rb");

	void* buffer = malloc_c(size);

	size_t read_size = fread_c(buffer, 1, size, stream);

	fclose_c(stream);

	return buffer;
}

void verify(void) {
	// TODO
	return;
}

model* loadNetwork() {
	model* network = (model*)malloc_c(sizeof(model));
	network->ptr = (float*)readByte(network_file, model_size);
	network->weights = (float**)malloc_c(sizeof(float*) * 21);
	network->biases = (float**)malloc_c(sizeof(float*) * 21);

	float* network_ptr = network->ptr;
	for (int i = 0; i < 21; ++i) {
		// maxpooling layer
		if (i == 2 || i == 5 || i == 9 || i == 13 || i == 17) {
			network->weights[i] = NULL;
			network->biases[i] = NULL;
		}
		// convolution layer
		else if (i < 18) {
			network->weights[i] = network_ptr;
			network_ptr += (9 * WIDTHS[i][0] * WIDTHS[i][1]);

			network->biases[i] = network_ptr;
			network_ptr += WIDTHS[i][1];
		}
		// fc layer
		else {
			network->weights[i] = network_ptr;
			network_ptr += (WIDTHS[i][0] * WIDTHS[i][1]);

			network->biases[i] = network_ptr;
			network_ptr += WIDTHS[i][1];
		}
	}

	return network;
}

void unloadNetwork(model* model_) {
	free_c(model_->weights);
	free_c(model_->biases);
	free_c(model_->ptr);
}

images* loadImages(const size_t num_images) {
	const size_t num_pixels = 32 * 32 * 3;
	const size_t image_size = sizeof(float) * num_pixels;

	images* image = (images*)malloc_c(sizeof(images));
	image->count = num_images;
	image->ptr = (float*)readByte(image_file, image_size * image->count);
	image->at = (float**)malloc_c(sizeof(float*) * image->count);

	for (unsigned int i = 0; i < image->count; ++i)
		image->at[i] = image->ptr + (i * num_pixels);

	return image;
}

void unloadImages(images* images_) {
	free_c(images_->at);
	free_c(images_->ptr);
}