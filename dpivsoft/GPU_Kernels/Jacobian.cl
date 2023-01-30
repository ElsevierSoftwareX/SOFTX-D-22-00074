KERNEL void Jacobian(GLOBAL_MEM float *output_dx,
GLOBAL_MEM float *output_dy,
GLOBAL_MEM float *input,
GLOBAL_MEM int *x,
GLOBAL_MEM int *y,
GLOBAL_MEM int *data)
{
    int no_boxes_x = data[4];
    int no_boxes_y = data[5];

    const SIZE_T i = get_global_id(0);

    int dx = x[i+1]-x[i];
    int dy = y[i+no_boxes_x]-y[i];

    int pos_y = i/no_boxes_x;
    int pos_x = i%no_boxes_x;

    if(pos_x>0 && pos_x<no_boxes_x-1)     {
        output_dx[i] = (input[i+1]-input[i-1])/(2*dx);
    }else if(pos_x == 0){
        output_dx[i] = (input[i+1]-input[i])/dx;
    }
    else if(pos_x == no_boxes_x-1){
        output_dx[i] = (input[i]-input[i-1])/(x[i]-x[i-1]);
    }

    if(pos_y>0 && pos_y<no_boxes_y-1){
        output_dy[i] = (input[i+no_boxes_x]-input[i-no_boxes_x])/(2*dy);
    }
    else if (pos_y == 0){
        output_dy[i] = (input[i+no_boxes_x]-input[i])/dy;
    }
    else if (pos_y == no_boxes_y-1){
        output_dy[i] = (input[i]-input[i-no_boxes_x])/(y[i]-y[i-no_boxes_x]);
    }
}
