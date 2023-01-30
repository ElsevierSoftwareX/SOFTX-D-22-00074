// Apply gaussian filter to the image
KERNEL void gaussian_filter(GLOBAL_MEM float *output,
GLOBAL_MEM float *input,
GLOBAL_MEM float *filter,
GLOBAL_MEM int *data)
{
    int w = data[0];  // width
    int h = data[1];  // height

    // size of the kernel
    int size_K = data[9]*4+1;
    int lim = size_K/2;
    int pos_k = size_K*lim+size_K/2;  //center position of the kernel

    const SIZE_T pos = get_global_id(0);

    int pos_y = pos/w;
    int pos_x = pos%w;

    //1 Initialize variable
    float total = 0;

    // Apply the filter to each pixel
    for (int j = -lim; j<=lim; j++)
    {
        for (int i = -lim; i<=lim; i++)
        {
            if(pos_x+i<1 || pos_x+i>=w-1 || pos_y+j<1 || pos_y+j>=h-1)
            {}
            else
            {
                total = total + input[pos + j*w + i] *
                    filter[pos_k + j*size_K + i];
            }

        }
    }
    // Write the results on the output pixel
    output[pos] = total;
}
