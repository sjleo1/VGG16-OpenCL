__kernel void conv(
	__global float *__private input,
	__global float *__private output,
	__global float *__private weight,
	__global float *__private bias,
	__private const unsigned short in_width,
	__private const unsigned short out_width,
	__private const unsigned char res,
	__local float *input_buffer
) {
	__private unsigned char out_y = get_global_id(1);
	__private unsigned char out_x = get_global_id(0);

	__private float filter[9];
	__private float output_buffer = 0.0f;

	weight += get_global_id(2) * in_width * 9;
	bias += get_global_id(2);
	output += get_global_id(2) * res * res \
			+ get_global_id(1) * res \
			+ get_global_id(0);

	for (unsigned short in_chan = 0; in_chan < in_width; ++in_chan) {
		input_buffer[get_local_id(1) * res + get_local_id(0)] = \
			input[get_global_id(1) * res + get_global_id(0)];
		for (unsigned char w = 0; w < 9; ++w) {
			filter[w] = weight[w];
		}
		//
		barrier(CLK_LOCAL_MEM_FENCE);

		for (unsigned char filter_y = 0; filter_y < 3; ++filter_y) {
			for (unsigned char filter_x = 0; filter_x < 3; ++filter_x) {
				unsigned char in_y = out_y + filter_y - 1;
				unsigned char in_x = out_x + filter_x - 1;

				if (in_y >= 0 && in_y < res && in_x >= 0 && in_x < res) {
					output_buffer += filter[filter_y * 3 + filter_x] \
						* input_buffer[in_y * res + in_x];
				}
			}
		}

		input += res * res;
		weight += 9;
	}

	output_buffer += *bias;
	if (output_buffer < 0.0f) {
		output_buffer = 0.0f;
	}

	*output = output_buffer;

	return;
}