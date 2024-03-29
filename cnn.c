#include <stdio.h>
#include <math.h>
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

void verify(const result* output) {
	const size_t num_images = output->count;

	result* answer = loadResult(num_images, true);

	size_t cnt_wrong = 0;
	int* wrong_answers = (int*)malloc_c(sizeof(int) * num_images);
	for (unsigned int i = 0; i < num_images; ++i) {
		double diff = fabs(output->confs[i] - answer->confs[i]);

		if (!(output->labels[i] == answer->labels[i] && diff < 0.01))
			wrong_answers[cnt_wrong++] = i;
	}

	printf("Accuracy: %.2lf%%\n", 100.0 * (double)(num_images - cnt_wrong) / (double)num_images);
	if (cnt_wrong) {
		printf("%zu wrong answers:\n", cnt_wrong);
		printf("===================================================\n");
		printf("Image       Category #              Confidence\n");
		printf("  #     Expected - Result       Expected - Result\n");
		printf("===================================================\n");
		for (unsigned int i = 0; i < cnt_wrong; ++i) {
			int idx = wrong_answers[i];
			printf(" % 4d         %2d - %-2d           %8f - %-8f\n", idx,
				answer->labels[idx], output->labels[idx],
				answer->confs[idx], output->confs[idx]);
		}
		printf("===================================================\n");
	}

	double time_spent = (double)(output->end_time - output->start_time) / CLOCKS_PER_SEC;
	printf("Elapsed time:           %.3lf seconds/%zu images\n", time_spent, num_images);
	printf("Elapsed time per image: %.5lf seconds/image\n", time_spent / (double)num_images);

	free_c(wrong_answers);
	unloadResult(answer);

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

result* loadResult(const size_t num_images, bool load_answer) {
	result* result_ = (result*)malloc_c(sizeof(result));
	result_->count = num_images;
	result_->labels = (int*)malloc_c(sizeof(int) * result_->count);
	result_->confs = (float*)malloc_c(sizeof(float) * result_->count);

	if (load_answer) {
		FILE* fp_answer = fopen_c(answer_file, "rb");

		for (unsigned int i = 0; i < result_->count; ++i) {
			size_t size_lbls = fread_c(&(result_->labels[i]), sizeof(int), 1, fp_answer);
			size_t size_confs = fread_c(&(result_->confs[i]), sizeof(float), 1, fp_answer);
		}

		fclose_c(fp_answer);
	}

	return result_;
}

extern void unloadResult(result* result_) {
	free_c(result_->labels);
	free_c(result_->confs);
	free_c(result_);
}