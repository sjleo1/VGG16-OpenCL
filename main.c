#include <stdio.h>

#include "cnn.h"
#include "customlib.h"

extern void parallel(const images*, const model*, int[], float[]);

void runInference() {
	const size_t n = 10;
	images* input = loadImages(n);
	model* network = loadNetwork();
	int* labels = malloc_c(sizeof(int) * n);
	float* conf = malloc_c(sizeof(float) * n);

	parallel(input, network, labels, conf);

	printf("\nResult:\n# Name       Confidence\n");
	for (unsigned int i = 0; i < n; ++i)
		printf("%d %-10s %f\n", labels[i], CLASS_NAME[labels[i]], conf[i]);

	unloadImages(input);
	unloadNetwork(network);
	free_c(labels);
	free_c(conf);
}


int main(int argc, char* argv[]) {
	runInference();

	return 0;
}