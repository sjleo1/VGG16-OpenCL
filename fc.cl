__kernel void fc(
	const __global float *__private input,
	__global float *__private output,
	const __global float *__private weight,
	const __global float *__private bias,
	const __private unsigned short num_innodes,
	const __private unsigned short num_outnodes,
	__local float *__private sub_block
) {
	input += get_global_id(0);
	output += get_global_id(1);
	weight += get_global_id(1) * num_innodes + get_global_id(0);
	bias += get_global_id(1);

	sub_block[get_global_id(0)] = *input * *weight;

	barrier(CLK_LOCAL_MEM_FENCE);

	for (unsigned short divider = num_innodes / 2; divider > 0; divider >>= 1) {
		if (get_global_id(0) < divider)
			sub_block[get_global_id(0)] += sub_block[get_global_id(0) + divider];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (get_global_id(0) == 0) {
		__private float sum = sub_block[0] + *bias;
		
		if (sum < 0.0f)
			*output = 0.0f;
		else
			*output = sum;
	}

	return;
}
