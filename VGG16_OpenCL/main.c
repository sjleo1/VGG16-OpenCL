#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>
#include "cnn.h"

void* readFile(const char* file_name, int nbytes);
void compare(const char* filename, int num_of_image);


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

const int INPUT_DIM[] = {
	3,
	64,
	64,

	64,
	128,
	128,

	128,
	256,
	256,
	256,

	256,
	512,
	512,
	512,

	512,
	512,
	512,
	512,

	512,
	512,
	512
};

const int OUTPUT_DIM[] = {
	64,
	64,
	64,

	128,
	128,
	128,

	256,
	256,
	256,
	256,

	512,
	512,
	512,
	512,

	512,
	512,
	512,
	512,

	512,
	512,
	10
};

const int NBYN[] = {
	32, 32,
	16,

	16, 16,
	8,

	8, 8, 8,
	4,

	4, 4, 4,
	2,

	2, 2, 2,
	1,

	1, 1, 1
};

int main(int argc, char* argv[]) {
	if (argc != 3) {
		perror("Error while getting arguments.\n");
		exit(1);
	}
	if (strcmp("answer.txt", argv[2]) == 0) {
		perror("'answer.txt' is an unauthorized name.\n");
		exit(1);
	}

	int num_of_image = atoi(argv[1]);
	if (num_of_image < 0 || num_of_image > 10000) {
		perror("The number of images should not be less than zero or greater than 10000.\n");
		exit(1);
	}

	float* images = (float*)readFile("images.bin", sizeof(float) * 32 * 32 * 3 * num_of_image);
	float* network = (float*)readFile("network.bin", 60980520);
	int* labels = (int*)malloc(sizeof(int) * num_of_image);
	float* confidences_par = (float*)malloc(sizeof(float) * num_of_image);
	float* confidences_seq = (float*)malloc(sizeof(float) * num_of_image);
	float** filters_and_biases = (float**)malloc(sizeof(float*) * 32);
	float* network_ptr = network;

	for (int i = 0, k = 0; i < 21; i++)
		if (i == 2 || i == 5 || i == 9 || i == 13 || i == 17)
			continue;
		else if (i < 18) {
			filters_and_biases[k++] = network_ptr;
			network_ptr += (9 * INPUT_DIM[i] * OUTPUT_DIM[i]);
			filters_and_biases[k++] = network_ptr;
			network_ptr += OUTPUT_DIM[i];
		}
		else {
			filters_and_biases[k++] = network_ptr;
			network_ptr += (INPUT_DIM[i] * OUTPUT_DIM[i]);
			filters_and_biases[k++] = network_ptr;
			network_ptr += OUTPUT_DIM[i];
		}

	// OpenCL
	parallelCNNInit();
	time_t start, end;
	start = clock();
	parallelCNN(images, filters_and_biases, labels, confidences_par, num_of_image);
	end = clock();
	printf("Elapsed time (OpenCL): %.2f sec.\n", (double)(end - start) / CLK_TCK);

	// Sequential
	sequentialCNN(images, network, labels, confidences_seq, num_of_image);


	int* labels_ans = (int*)readFile("labels.bin", sizeof(int) * num_of_image);
	double acc = 0.0;

	FILE* fp = fopen(argv[2], "w");
	for (int i = 0; i < num_of_image; i++) {
		fprintf(fp, "Image %04d : %d : %-10s\t%f\n", i, labels[i], CLASS_NAME[labels[i]], confidences[i]);
		if (labels[i] == labels_ans[i])
			acc++;
	}
	fprintf(fp, "Accuracy: %f\n", acc / num_of_image);
	fclose(fp);

	compare(argv[2], num_of_image);

	free(images);
	free(network);
	free(labels);
	free(confidences);
	free(labels_ans);

	return 0;
}

int _main(int argc, char* argv[]) {
	if (argc != 3) {
		perror("Error while getting arguments.\n");
		exit(1);
	}
	if (strcmp("answer.txt", argv[2]) == 0) {
		perror("'answer.txt' is an unauthorized name.\n");
		exit(1);
	}

	int num_of_image = atoi(argv[1]);
	if (num_of_image < 0 || num_of_image > 10000) {
		perror("The number of images should not be less than zero or greater than 10000.\n");
		exit(1);
	}

	float* images = (float*)readFile("images.bin", sizeof(float) * 32 * 32 * 3 * num_of_image);
	float* network = (float*)readFile("network.bin", 60980520);
	int* labels = (int*)malloc(sizeof(int) * num_of_image);
	float* confidences = (float*)malloc(sizeof(float) * num_of_image);
	float** filters_and_biases = (float**)malloc(sizeof(float*) * 32);
	float* network_ptr = network;

	for (int i = 0, k = 0; i < 21; i++)
		if (i == 2 || i == 5 || i == 9 || i == 13 || i == 17)
			continue;
		else if (i < 18) {
			filters_and_biases[k++] = network_ptr;
			network_ptr += (9 * INPUT_DIM[i] * OUTPUT_DIM[i]);
			filters_and_biases[k++] = network_ptr;
			network_ptr += OUTPUT_DIM[i];
		}
		else {
			filters_and_biases[k++] = network_ptr;
			network_ptr += (INPUT_DIM[i] * OUTPUT_DIM[i]);
			filters_and_biases[k++] = network_ptr;
			network_ptr += OUTPUT_DIM[i];
		}

	// OpenCL
	parallelCNNInit();
	time_t start, end;
	start = clock();
	parallelCNN(images, filters_and_biases, labels, confidences, num_of_image);
	end = clock();
	printf("Elapsed time (OpenCL): %.2f sec.\n", (double)(end - start) / CLK_TCK);

	// Sequential
	start = clock();
	sequentialCNN(images, network, labels, confidences, num_of_image);
	end = clock();
	printf("Elapsed time (OpenCL): %.2f sec.\n", (double)(end - start) / CLK_TCK);


	int* labels_ans = (int*)readFile("labels.bin", sizeof(int) * num_of_image);
	double acc = 0.0;

	FILE* fp = fopen(argv[2], "w");
	for (int i = 0; i < num_of_image; i++) {
		fprintf(fp, "Image %04d : %d : %-10s\t%f\n", i, labels[i], CLASS_NAME[labels[i]], confidences[i]);
		if (labels[i] == labels_ans[i])
			acc++;
	}
	fprintf(fp, "Accuracy: %f\n", acc / num_of_image);
	fclose(fp);

	compare(argv[2], num_of_image);

	free(images);
	free(network);
	free(labels);
	free(confidences);
	free(labels_ans);

	return 0;
}

void* readFile(const char* file_name, int nbytes) {
	void* buf = malloc(nbytes);
	if (buf == NULL) {
		perror("Error while allocating memory.\n");
		exit(1);
	}

	FILE* fp = fopen(file_name, "rb");
	if (fp == NULL) {
		perror("Error while opening %s.\n", file_name);
		exit(1);
	}

	int retv = fread(buf, 1, nbytes, fp);
	if (retv != nbytes) {
		perror("Error while reading.\n");
		exit(1);
	}

	if (fclose(fp) != 0) {
		perror("Error while closing.\n");
		exit(1);
	}

	return buf;
}

void compare(const char* file_name, int num_of_image) {
	FILE* fp1 = fopen("answer.txt", "r");
	if (fp1 == NULL) {
		perror("Failed to open 'answer.txt'.\n");
		exit(1);
	}
	FILE* fp2 = fopen(file_name, "r");
	if (fp2 == NULL) {
		perror("Error while opening %s.\n", file_name);
		exit(1);
	}
	int retv;
	int* correct_class = (int*)malloc(sizeof(int) * num_of_image);
	int* your_class = (int*)malloc(sizeof(int) * num_of_image);
	float* correct_conf = (float*)malloc(sizeof(float) * num_of_image);
	float* your_conf = (float*)malloc(sizeof(float) * num_of_image);


	printf("\n");
	for (int i = 0; i < num_of_image; i++) {
		retv = fscanf(fp1, "Image %*4d : %d : %*10s\t%f\n", correct_class + i, correct_conf + i);
		if (retv = 0) {
			perror("Error occurred during fscanf.");
			exit(1);
		}
		retv = fscanf(fp2, "Image %*4d : %d : %*10s\t%f\n", your_class + i, your_conf + i);
		if (retv = 0) {
			perror("Error occurred during fscanf.");
			exit(1);
		}
		//printf("%d : %d, %f %f\n", correct_class[i], your_class[i], correct_conf[i], your_conf[i]);
		if (correct_class[i] != your_class[i] || fabs(correct_conf[i] - your_conf[i]) > 0.01) {
			printf("Images %04d ", i);
			printf("%10s : %f is correct. but your answer is ", CLASS_NAME[correct_class[i]], correct_conf[i]);
			printf("%10s : %f\n", CLASS_NAME[your_class[i]], your_conf[i]);
		}
	}

	free(correct_class);
	free(your_class);
	free(correct_conf);
	free(your_conf);

	return;
}