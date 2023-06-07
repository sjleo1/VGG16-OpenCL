#pragma once

#ifdef _MSC_VER
#pragma warning (disable: 4996)
#endif

#include <time.h>
#include <stdbool.h>

typedef struct Model {
	float* ptr;
	float** weights;
	float** biases;
} model;

typedef struct Images {
	float* ptr;
	size_t count;
	float** at;
} images;

typedef struct Result {
	size_t count;
	int* labels;
	float* confs;
	clock_t start_time;
	clock_t end_time;
} result;

extern const size_t model_size;
extern const char image_file[];
extern const char label_file[];
extern const char network_file[];
extern const char answer_file[];
extern const char* CLASS_NAME[];
extern const size_t WIDTHS[][2];
extern const size_t RES[];

extern void* readByte(const char*, size_t);
extern void verify(const result*);
extern model* loadNetwork();
extern void unloadNetwork(model*);
extern images* loadImages(const size_t);
extern void unloadImages(images*);
extern result* loadResult(const size_t, bool);
extern void unloadResult(result*);