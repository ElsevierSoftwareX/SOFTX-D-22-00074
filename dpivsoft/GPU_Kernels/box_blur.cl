KERNEL void box_blur(GLOBAL_MEM float *output,
GLOBAL_MEM float *input,
GLOBAL_MEM int *data)
{
    int w = data[4];
    int h = data[5];

    const SIZE_T i = get_global_id(0);

    int pos_y = i/w;
    int pos_x = i%w;

    if (pos_x == 0 || pos_x == w-1 || pos_y == 0 || pos_y == h-1){
        output[i] = input[i];
    }
    else{
        output[i] = (input[i+w-1]+input[i+w]+input[i+w+1]+input[i-1]+input[i]
                      +input[i+1]+input[i-w-1]+input[i-w]+input[i-w+1])/9;
    }
}
