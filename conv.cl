#define out_z get_global_id(2)
#define out_y get_global_id(1)
#define out_x get_global_id(0)

__kernel void conv_low(
	const __global float *__private input,
	__global float *__private output,
	const __global float *__private weight,
	const __global float *__private bias,
	const __private unsigned short in_width,
	const __private unsigned short out_width,
	const __private unsigned char res,
	__local float *__private input_buffer
) {
	__private float output_buffers[WPT * WPT] = { 0.0f };
	__private float filter[9];

	output += out_z * (res * res) + out_y * res + out_x;
	weight += out_z * in_width * 9;
	bias += out_z;

	for (unsigned short in_chan = 0; in_chan < in_width; ++in_chan) {
		for (unsigned char dy = 0; dy < WPT; ++dy)
			for (unsigned char dx = 0; dx < WPT; ++dx)
				input_buffer[(out_y + dy) * res + out_x + dx] = \
					input[(out_y + dy) * res + out_x + dx];
		for (unsigned char w = 0; w < 9; ++w)
			filter[w] = weight[w];

		barrier(CLK_LOCAL_MEM_FENCE);

		for (unsigned char dy = 0; dy < WPT; ++dy) {
			for (unsigned char dx = 0; dx < WPT; ++dx) {
				for (unsigned char filter_y = 0; filter_y < 3; ++filter_y) {
					for (unsigned char filter_x = 0; filter_x < 3; ++filter_x) {
						unsigned char in_y = out_y + dy + filter_y - 1;
						unsigned char in_x = out_x + dx + filter_x - 1;

						if (in_y >= 0 && in_y < res && in_x >= 0 && in_x < res)
							output_buffers[dy * WPT + dx] += \
								filter[filter_y * 3 + filter_x] \
								* input_buffer[in_y * res + in_x];
					}
				}
			}
		}

		input += res * res;
		weight += 9;

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	for (unsigned char dy = 0; dy < WPT; ++dy)
		for (unsigned char dx = 0; dx < WPT; ++dx) {
			output_buffers[dy * WPT + dx] += *bias;
			if (output_buffers[dy * WPT + dx] < 0.0f)
				output[dy * res + dx] = 0.0f;
			else
				output[dy * res + dx] = output_buffers[dy * WPT + dx];
		}

	return;
}


__kernel void conv_high(
	const __global float *__private input,
	__global float *__private output,
	const __global float *__private weight,
	const __global float *__private bias,
	const __private unsigned short in_width,
	const __private unsigned short out_width,
	const __private unsigned char res,
	__local float *__private input_buffer
) {
	__private float output_buffer = 0.0f;
	__private float filter[9];

	output += out_z * (res * res) + out_y * res + out_x;
	weight += out_z * in_width * 9;
	bias += out_z;

	for (unsigned short in_chan = 0; in_chan < in_width; ++in_chan) {
		input_buffer[get_local_id(1) * res + get_local_id(0)] = \
			input[get_global_id(1) * res + get_global_id(0)];
		for (unsigned char w = 0; w < 9; ++w)
			filter[w] = weight[w];

		barrier(CLK_LOCAL_MEM_FENCE);

		for (unsigned char filter_y = 0; filter_y < 3; ++filter_y) {
			for (unsigned char filter_x = 0; filter_x < 3; ++filter_x) {
				unsigned char in_y = out_y + filter_y - 1;
				unsigned char in_x = out_x + filter_x - 1;

				if (in_y >= 0 && in_y < res && in_x >= 0 && in_x < res)
					output_buffer += \
						filter[filter_y * 3 + filter_x] \
						* input_buffer[in_y * res + in_x];
			}
		}

		input += res * res;
		weight += 9;

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	output_buffer += *bias;
	if (output_buffer < 0.0f)
		*output = 0.0f;
	else
		*output = output_buffer;

	return;
}
