KERNEL void Interpolation(GLOBAL_MEM float *output,
GLOBAL_MEM float *input,
GLOBAL_MEM int *x2,
GLOBAL_MEM int *y2,
GLOBAL_MEM int *x1,
GLOBAL_MEM int *y1,
GLOBAL_MEM int *data)
{
    int no_boxes_x = data[4];

    const SIZE_T i = get_global_id(0);

    int dx = x1[1]-x1[0];
    int dy = y1[no_boxes_x]-y1[0];

    int pos_x = (x2[i]-x1[0])/dx;
    int pos_y = (y2[i]-y1[0])/dy;

    int pos = pos_y*no_boxes_x+pos_x;

    float temp1 = (x2[i]-x1[pos]+0.0)/dx;
    float temp2 = (y2[i]-y1[pos]+0.0)/dy;

    output[i] = input[pos] * (1-temp1) * (1-temp2)
       + input[pos+1] * (temp1) * (1-temp2)
       + input[pos+1+no_boxes_x] * temp1 * temp2
       + input[pos+no_boxes_x] * (1-temp1) * (temp2);
}
