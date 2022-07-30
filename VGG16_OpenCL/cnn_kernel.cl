// LOCAL_WH < NbyN
__kernel void convolution_ker_A(__global float* inputs,
								__global float* filters,
								__global float* biases,
								__global float* outputs,
								__local float* input,
								__local float* filter,
								__private const short indim,
								__private const short outdim,
								__private const char NbyN) {
	//__private const char id[2][3] = { { get_local_id(0), get_local_id(1), get_local_id(2) }, { get_global_id(0), get_global_id(1), get_global_id(2) } };
	__private const int in_batch = (NbyN * NbyN * indim * DOP * (get_global_id(2) / (outdim / DEPTH)));
	__private const int out_batch = (NbyN * NbyN * outdim * DOP * (get_global_id(2) / (outdim / DEPTH)));
	__private const short out_z_limit = (get_global_id(2) % (outdim / DEPTH)) * DEPTH + DEPTH;
	__private char x, y;

	__global float* pinputs, * poutputs;

	for (short out_z = out_z_limit - DEPTH; out_z < out_z_limit; out_z++) {
		poutputs = outputs + out_batch + NbyN * NbyN * out_z;
		for (char dop = 0; dop < DOP; dop++) {
			for (y = get_local_id(1); y < NbyN; y += LOCAL_WH)
				for (x = get_local_id(0); x < NbyN; x += LOCAL_WH)
					poutputs[NbyN * y + x] = 0.0f;
			poutputs += NbyN * NbyN * outdim;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for (short in_z = 0; in_z < indim; in_z++) {
		pinputs = inputs + in_batch + NbyN * NbyN * in_z;
		for (short out_z = out_z_limit - DEPTH; out_z < out_z_limit; out_z++) {
			poutputs = outputs + out_batch + NbyN * NbyN * out_z;
			if (get_local_id(1) < 3 && get_local_id(0) < 3)
				filter[9 * get_local_id(2) + 3 * get_local_id(1) + get_local_id(0)] = \
				filters[9 * indim * out_z + 9 * in_z + 3 * get_local_id(1) + get_local_id(0)];
			for (char dop = 0; dop < DOP; dop++) {
				for (y = get_local_id(1); y < NbyN; y += LOCAL_WH)
					for (x = get_local_id(0); x < NbyN; x += LOCAL_WH)
						input[NbyN * y + x] = pinputs[NbyN * y + x];
				barrier(CLK_LOCAL_MEM_FENCE);//
				for (y = get_local_id(1); y < NbyN; y += LOCAL_WH)
					for (x = get_local_id(0); x < NbyN; x += LOCAL_WH) {
						__private float acc = 0.0f;
						for (char f_row = 0; f_row < 3; f_row++) {
							__private const char row = y + f_row - 1;
							for (char f_col = 0; f_col < 3; f_col++) {
								__private const char col = x + f_col - 1;
								if (row >= 0 && row < NbyN && col >= 0 && col < NbyN)
									acc += input[NbyN * row + col] * filter[9 * get_local_id(2) + 3 * f_row + f_col];
							}
						}
						poutputs[NbyN * y + x] += acc;
					}
				barrier(CLK_LOCAL_MEM_FENCE);//
				pinputs += NbyN * NbyN * indim;
				poutputs += NbyN * NbyN * outdim;
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}

	for (short out_z = out_z_limit - DEPTH; out_z < out_z_limit; out_z++) {
		__private const float bias = biases[out_z];
		poutputs = outputs + out_batch + NbyN * NbyN * out_z;
		for (char dop = 0; dop < DOP; dop++) {
			for (y = get_local_id(1); y < NbyN; y += LOCAL_WH)
				for (x = get_local_id(0); x < NbyN; x += LOCAL_WH) {
					if ((poutputs[NbyN * y + x] + bias) < 0.0f)
						poutputs[NbyN * y + x] = 0.0f;
					else
						poutputs[NbyN * y + x] += bias;
				}
			poutputs += NbyN * NbyN * outdim;
		}
	}

	return;
}

// LOCAL_WH == NbyN
__kernel void convolution_ker_B(__global float* inputs,
								__global float* filters,
								__global float* biases,
								__global float* outputs,
								__local float* input,
								__local float* filter,
								__private const short indim,
								__private const short outdim,
								__private const char NbyN) {
	//__private const char id_2[2] = { get_local_id(2), get_global_id(2) };
	__private const int in_batch = (NbyN * NbyN * indim * DOP * (get_global_id(2) / (outdim / DEPTH)));
	__private const int out_batch = (NbyN * NbyN * outdim * DOP * (get_global_id(2) / (outdim / DEPTH)));
	__private const short out_z_limit = (get_global_id(2) % (outdim / DEPTH)) * DEPTH + DEPTH;
	__private const char y = get_global_id(1);
	__private const char x = get_global_id(0);

	__global float* pinputs, * poutputs;

	for (short out_z = out_z_limit - DEPTH; out_z < out_z_limit; out_z++) {
		poutputs = outputs + out_batch + NbyN * NbyN * out_z;
		for (char dop = 0; dop < DOP; dop++)
			poutputs[NbyN * NbyN * outdim * dop + NbyN * y + x] = 0.0f;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for (short in_z = 0; in_z < indim; in_z++) {
		pinputs = inputs + in_batch + NbyN * NbyN * in_z;
		for (short out_z = out_z_limit - DEPTH; out_z < out_z_limit; out_z++) {
			poutputs = outputs + out_batch + NbyN * NbyN * out_z;
			if (y < 3 && x < 3)
				filter[9 * get_local_id(2) + 3 * y + x] = filters[9 * indim * out_z + 9 * in_z + 3 * y + x];
			for (char dop = 0; dop < DOP; dop++) {
				input[NbyN * y + x] = pinputs[NbyN * NbyN * indim * dop + NbyN * y + x];
				__private float acc = 0.0f;
				for (char f_row = 0; f_row < 3; f_row++) {
					__private const char row = y + f_row - 1;
					for (char f_col = 0; f_col < 3; f_col++) {
						__private const char col = x + f_col - 1;
						if (row >= 0 && row < NbyN && col >= 0 && col < NbyN)
							acc += input[NbyN * row + col] * filter[9 * get_local_id(2) + 3 * f_row + f_col];
					}
				}
				poutputs[NbyN * NbyN * outdim * dop + NbyN * y + x] += acc;
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}

	for (short out_z = out_z_limit - DEPTH; out_z < out_z_limit; out_z++) {
		poutputs = outputs + out_batch + NbyN * NbyN * out_z;
		for (char dop = 0; dop < DOP; dop++) {
			if ((poutputs[NbyN * NbyN * outdim * dop + NbyN * y + x] + biases[out_z]) < 0.0f)
				poutputs[NbyN * NbyN * outdim * dop + NbyN * y + x] = 0.0f;
			else
				poutputs[NbyN * NbyN * outdim * dop + NbyN * y + x] += biases[out_z];
		}
	}

	return;
}

// NbyN < 3
__kernel void convolution_ker_C(__global float* inputs,
								__global float* filters,
								__global float* biases,
								__global float* outputs,
								__local float* input,
								__local float* filter,
								__private const short indim,
								__private const short outdim,
								__private const char NbyN) {
	//__private const char id_2[2] = { get_local_id(2), get_global_id(2) };
	__private const int in_batch = (NbyN * NbyN * indim * DOP * (get_global_id(2) / (outdim / DEPTH)));
	__private const int out_batch = (NbyN * NbyN * outdim * DOP * (get_global_id(2) / (outdim / DEPTH)));
	__private const short out_z_limit = (get_global_id(2) % (outdim / DEPTH)) * DEPTH + DEPTH;
	__private const char y = get_global_id(1);
	__private const char x = get_global_id(0);

	__global float* pinputs, * poutputs;

	for (short out_z = out_z_limit - DEPTH; out_z < out_z_limit; out_z++) {
		poutputs = outputs + out_batch + NbyN * NbyN * out_z;
		for (char dop = 0; dop < DOP; dop++)
			poutputs[NbyN * NbyN * outdim * dop + NbyN * y + x] = 0.0f;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for (short in_z = 0; in_z < indim; in_z++) {
		pinputs = inputs + in_batch + NbyN * NbyN * in_z;
		for (short out_z = out_z_limit - DEPTH; out_z < out_z_limit; out_z++) {
			poutputs = outputs + out_batch + NbyN * NbyN * out_z;
			filter[9 * get_local_id(2) + 2 * y + x] = filters[9 * indim * out_z + 9 * in_z + 2 * y + x];
			filter[9 * get_local_id(2) + 2 * y + x + 5] = filters[9 * indim * out_z + 9 * in_z + 2 * y + x + 5];
			if (x == 0 && y == 0)
				filter[9 * get_local_id(2) + 4] = filters[9 * indim * out_z + 9 * in_z + 4];
			for (char dop = 0; dop < DOP; dop++) {
				input[NbyN * y + x] = pinputs[NbyN * NbyN * indim * dop + NbyN * y + x];
				__private float acc = 0.0f;
				for (char f_row = 0; f_row < 3; f_row++) {
					__private const char row = y + f_row - 1;
					for (char f_col = 0; f_col < 3; f_col++) {
						__private const char col = x + f_col - 1;
						if (row >= 0 && row < NbyN && col >= 0 && col < NbyN)
							acc += input[NbyN * row + col] * filter[9 * get_local_id(2) + 3 * f_row + f_col];
					}
				}
				poutputs[NbyN * NbyN * outdim * dop + NbyN * y + x] += acc;
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}

	for (short out_z = out_z_limit - DEPTH; out_z < out_z_limit; out_z++) {
		poutputs = outputs + out_batch + NbyN * NbyN * out_z;
		for (char dop = 0; dop < DOP; dop++) {
			if ((poutputs[NbyN * NbyN * outdim * dop + NbyN * y + x] + biases[out_z]) < 0.0f)
				poutputs[NbyN * NbyN * outdim * dop + NbyN * y + x] = 0.0f;
			else
				poutputs[NbyN * NbyN * outdim * dop + NbyN * y + x] += biases[out_z];
		}
	}

	return;
}

__kernel void max_pooling_ker(__global float* inputs, __global float* outputs, __private const short dim, __private char NbyN) {
	__global float* pinputs = inputs + (NbyN * NbyN * dim * (get_global_id(2) / dim)) + NbyN * NbyN * (get_global_id(2) % dim) + NbyN * (get_local_id(1) * 2) + (get_local_id(0) * 2);

	__private float max = pinputs[0];
	if (max < pinputs[1])
		max = pinputs[1];
	if (max < pinputs[NbyN])
		max = pinputs[NbyN];
	if (max < pinputs[NbyN + 1])
		max = pinputs[NbyN + 1];
	
	NbyN /= 2;
	outputs[(NbyN * NbyN * dim * (get_global_id(2) / dim)) + NbyN * NbyN * (get_global_id(2) % dim) + NbyN * get_local_id(1) + get_local_id(0)] = max;

	return;
}

// outdim == 512
__kernel void fc_layer_ker_A(__global float* inputs,
							__global float* filters,
							__global float* biases,
							__global float* outputs,
							__local float* input,
							__local float* filter) {
	__private const char num_of_tiles = 512 / LOCAL_WH;
	__private const char out_neuron = get_global_id(1);
	__private const bool shouldCalculate = get_global_id(0) < DOP;
	__private const short tile_ind = LOCAL_WH * get_local_id(1) + get_local_id(0);

	__global float* pinputs = inputs + 512 * get_global_id(0) + get_local_id(1);
	__global float* pfilters = filters + 512 * out_neuron + get_local_id(0);
	__global float* poutputs = outputs + 512 * get_global_id(0) + out_neuron;

	if (shouldCalculate)
		*poutputs = 0.0f;
	
	for (char tile = 0; tile < num_of_tiles; tile++) {
		__private float acc = 0.0f;

		filter[tile_ind] = pfilters[LOCAL_WH * tile];

		if (shouldCalculate)
			input[tile_ind] = pinputs[LOCAL_WH * tile];

		barrier(CLK_LOCAL_MEM_FENCE);

		if (shouldCalculate) {
			for (char in_neuron = 0; in_neuron < LOCAL_WH; in_neuron++)
				acc += filter[LOCAL_WH * get_local_id(1) + in_neuron] * input[LOCAL_WH * in_neuron + get_local_id(0)];
			*poutputs += acc;
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (shouldCalculate) {
		if ((*poutputs + biases[out_neuron]) < 0.0f)
			*poutputs = 0.0f;
		else
			*poutputs += biases[out_neuron];
	}
	
	return;
}

// outdim == 10
__kernel void fc_layer_ker_B(__global float* inputs,
							__global float* filters,
							__global float* biases,
							__global float* outputs,
							__local float* input,
							__local float* filter) {
	__private const char num_of_tiles = 512 / get_local_size(0);
	__private const char out_neuron = get_global_id(1) % 10;
	__private const bool shouldCalculate = (get_global_id(1) < 10 * BATCH && get_global_id(0) < DOP);
	__private const short tile_ind = LOCAL_WH * get_local_id(1) + get_local_id(0);
	
	__global float* pinputs = inputs + 512 * (DOP * (get_global_id(1) / 10) + get_global_id(0)) + get_local_id(1);
	__global float* pfilters = filters + 512 * out_neuron + get_local_id(0);
	__global float* poutputs = outputs + 10 * (DOP * (get_global_id(1) / 10) + get_global_id(0)) + out_neuron;

	if (shouldCalculate)
		*poutputs = 0.0f;

	for (char tile = 0; tile < num_of_tiles; tile++) {
		__private float acc = 0.0f;

		if (get_local_id(1) < 10)
			filter[tile_ind] = pfilters[LOCAL_WH * tile];

		if (get_global_id(0) < DOP)
			input[tile_ind] = pinputs[LOCAL_WH * tile];

		barrier(CLK_LOCAL_MEM_FENCE);

		if (shouldCalculate) {
			for (char in_neuron = 0; in_neuron < LOCAL_WH; in_neuron++)
				acc += filter[LOCAL_WH * out_neuron + in_neuron] * input[LOCAL_WH * in_neuron + get_local_id(0)];
			*poutputs += acc;
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (shouldCalculate) {
		if ((*poutputs + biases[out_neuron]) < 0.0f)
			*poutputs = 0.0f;
		else
			*poutputs += biases[out_neuron];
	}

	return;
}

__kernel void softmax_ker(	__global float* results,
							__local float* result,
							__global int* labels,
							__global float* confidences) {
	if (get_global_id(0) >= BATCH * DOP)
		return;
	
	__local float* presult = result + 10 * get_local_id(0);
	__private float max, sum = 0.0f;
	__private char max_index = 0;

	for (char i = 0; i < 10; i++)
		presult[i] = results[10 * get_global_id(0) + i];
	
	max = presult[0];

	for (char i = 1; i < 10; i++)
		if (max < presult[i]) {
			max = presult[i];
			max_index = i;
		}
	
	for (char i = 0; i < 10; i++)
		sum += exp(presult[i] - max);

	for (char i = 0; i < 10; i++)
		results[10 * get_global_id(0) + i] = exp(presult[i] - max) / (sum + 1e-7);

	labels[get_global_id(0)] = max_index;
	confidences[get_global_id(0)] = results[10 * get_global_id(0) + max_index];
	
	return;
}

// carriage return