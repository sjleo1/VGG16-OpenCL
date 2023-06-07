#include <stdio.h>
#include "cnn.h"
#include "customlib.h"

extern result* sequential(const images*, const model*);
extern result* parallel(const images*, const model*);

void runInference() {
	const size_t n = 10;

	images* input = loadImages(n);
	model* network = loadNetwork();
	result* output;

	int option;
	printf("0: Sequential\n1: Parallel\n-> ");
	scanf("%d", &option);

	if (option == 0) {
		output = sequential(input, network);
		verify(output);
		unloadResult(output);
	}
	else if (option == 1) {
		output = parallel(input, network);
		verify(output);
		unloadResult(output);
	}
	else
		printf("Invalid option");

	unloadImages(input);
	unloadNetwork(network);
}


int main(int argc, char* argv[]) {
	runInference();

	return 0;
}