// 2 by 2 long pipe
__kernel void convolution_ker(__global float* inputs,
                                __global float* filters,
                                __global float* biases,
                                __global float* outputs,
                                __local float* input,
                                __local float* filter,
                                __private const short indim,
                                __private const short outdim,
                                __private const short N) {
    //without tiling first
    __private float sum = 0.0f;
    __private short y = get_global_id(1);
    __private short x = get_global_id(0);


    for (short inneuron = 0; inneuron < indim; inneuron++) {
        for (short i = 0; i < 3; i++) {
            short f_y = y + i - 1;
            if (f_y >= 0 && f_y < get_global_size(1)) {
                for (short j = 0; j < 3; j++) {
                    short f_x = x + j - 1;
                    if (f_x >= 0 && f_x < get_global_size(0))
                        sum += inputs[inneuron * N * N + get_global_id(1) * N + get_global_id(0)] * filters[get_global_id(2) * 9 * indim + inneuron * 9 + i * 3 + j];
                }
            }
        }
    }

    outputs[get_global_id(2) * N * N + get_global_id(1) * N + get_global_id(0)] = sum;

    return;
}