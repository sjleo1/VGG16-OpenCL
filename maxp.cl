__kernel void maxp(
	const __global float *__private input,
	__global float *__private output,
	const __private short width,
	const __private unsigned char res
) {
	input += get_global_id(2) * (res * res * 4) \
		+ get_global_id(1) * (res * 4) \
		+ get_global_id(0) * 2;
	output += get_global_id(2) * (res * res) \
		+ get_global_id(1) * res \
		+ get_global_id(0);
		
	__private float max = 0.0f;
	
	if (max < *input)
		max = *input;
	
	input += 1;
	
	if (max < *input)
		max = *input;
	
	input += res * 2;
	
	if (max < *input)
		max = *input;
	
	input -= 1;
	
	if (max < *input)
		max = *input;
	
	*output = max;
	
	return;
}
