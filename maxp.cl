__kernel void maxp(
	__global float *input,
	__global float *output,
	__private const short width,
	__private const unsigned char res
) {
	__global float *__private read_from = input \
		+ get_global_id(2) * (res * res * 4) \
		+ get_global_id(1) * (res * 4) \
		+ get_global_id(0) * 2;
	__global float *__private write_to = output \
		+ get_global_id(2) * (res * res) \
		+ get_global_id(1) * res \
		+ get_global_id(0);
		
	__private float max = 0.0f;
	
	if (max < *read_from) {
		max = *read_from;
	}
	
	read_from += 1;
	
	if (max < *read_from) {
		max = *read_from;
	}
	
	read_from += res * 2;
	
	if (max < *read_from) {
		max = *read_from;
	}
	
	read_from -= 1;
	
	if (max < *read_from) {
		max = *read_from;
	}
	
	*write_to = max;
	
	return;
}