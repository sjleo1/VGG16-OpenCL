#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "customlib.h"

void* malloc_c(const size_t size) {
	void* buffer = malloc(size);

	if (buffer == NULL) {
		fprintf(stderr, "malloc failed: %s", strerror(errno));
		exit(errno);
	}

	return buffer;
}

void free_c(void* block) {
	free(block);
}

FILE* fopen_c(const char* file_name, const char* mode) {
	FILE* file = fopen(file_name, mode);

	if (file == NULL) {
		fprintf(stderr, "fopen failed: %s", strerror(errno));
		exit(errno);
	}

	return file;
}

size_t fread_c(void* buffer, size_t element_size, size_t element_count, FILE* stream) {
	size_t size = fread(buffer, element_size, element_count, stream);

	if (ferror(stream)) {
		fprintf(stderr, "fread failed: %s", strerror(errno));
		exit(errno);
	}

	return size;
}

void fclose_c(FILE* stream) {
	if (fclose(stream)) {
		fprintf(stderr, "fclose failed: %s", strerror(errno));
		exit(errno);
	}
}
