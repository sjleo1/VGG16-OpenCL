#pragma once

#ifdef _MSC_VER
#pragma warning (disable: 4996)
#endif

void* malloc_c(const size_t);
void free_c(void*);
FILE* fopen_c(const char*, const char*);
size_t fread_c(void*, size_t, size_t, FILE*);
void fclose_c(FILE*);
