#pragma once

#ifdef _MSC_VER
#pragma warning (disable: 4996)
#endif

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

extern const size_t model_size;
extern const char image_file[];
extern const char network_file[];
extern const char* CLASS_NAME[];
extern const size_t WIDTHS[][2];
extern const size_t RES[];

extern void* readByte(const char*, size_t);
extern void verify(void);
extern model* loadNetwork();
extern images* loadImages(const size_t);