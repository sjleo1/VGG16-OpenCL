__kernel void argmax(
    const __global float *__private arr,
    __global int *__private label,
    __global float *__private confidence
) {
    __private float arr_buffer[NUM_CLASSES];

    for (unsigned char i = 0; i < NUM_CLASSES; ++i)
        arr_buffer[i] = arr[i];

    __private float max = arr_buffer[0];
    __private unsigned char label_buffer = 0;
    __private float sum = 0.0f;
    
    for (unsigned char i = 0; i < NUM_CLASSES; ++i) {
        arr_buffer[i] = exp(arr_buffer[i]);

        if (max < arr_buffer[i]) {
            max = arr_buffer[i];
            label_buffer = i;
        }
        
        sum += arr_buffer[i];
    }

    *label = (int)label_buffer;
    *confidence = 1 / ((sum / max) + 1e-7f);

    return;
}
