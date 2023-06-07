#include <stdio.h>
#include "cnn.h"
#include "customlib.h"

extern result* sequential(const images*, const model*);

void runInference() {
	const size_t n = 10;

	images* input = loadImages(n);
	model* network = loadNetwork();

	result* output = sequential(input, network);

	verify(output);

	unloadImages(input);
	unloadNetwork(network);
	unloadResult(output);
}


#include "cnn.h"
#include "customlib.h"

extern result* parallel(const images*, const model*);

void runInference() {
	const size_t n = 10;

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