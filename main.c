#include <stdio.h>

int main(int argc, char* argv[]) {
	while (argc--) printf("%s\n", argv[argc]);
	return 0;
}