#include <stdio.h>

#include "cnn.h"
#include "customlib.h"

extern result* parallel(const images*, const model*);

void runInference() {
	const size_t n = 100;

	images* input = loadImages(n);
	model* network = loadNetwork();

	result* output = parallel(input, network);

	verify(output);

	unloadImages(input);
	unloadNetwork(network);
	unloadResult(output);
}


int main(int argc, char* argv[]) {
	runInference();

	return 0;
}