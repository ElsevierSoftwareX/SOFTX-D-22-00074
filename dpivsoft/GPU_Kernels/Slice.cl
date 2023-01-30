//Split image Kernel

KERNEL void Slice(GLOBAL_MEM float2 *subImg,
GLOBAL_MEM float *img,
GLOBAL_MEM int *box_origin_x,
GLOBAL_MEM int *box_origin_y,
GLOBAL_MEM int *data)
{
    int width = data[0];
    int box_size_x = data[2];
    int box_size_y = data[3];
    const SIZE_T i = get_global_id(0);

    int pos = i/(box_size_y*box_size_x);
    int pos_y = (i%(box_size_y*box_size_x))/box_size_x;
    int pos_x = (i%(box_size_y*box_size_x))%box_size_x;

    int k = (box_origin_y[pos]+pos_y)*width+box_origin_x[pos]+pos_x;

    subImg[i].x = img[k];
    subImg[i].y = 0;
}

