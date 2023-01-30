// Weighting Kernel

KERNEL void Weighting(GLOBAL_MEM float2 *subImg,
GLOBAL_MEM int *data)
{
    int width = data[0];
    int box_size_x = data[2];
    int box_size_y = data[3];
    int no_boxes_x = data[4];
    int no_boxes_y = data[5];
    const SIZE_T i = get_global_id(0);

    int pos = i/(box_size_y*box_size_x);
    float pos_y = (i%(box_size_y*box_size_x))/box_size_x;
    float pos_x = (i%(box_size_y*box_size_x))%box_size_x;

    float zeta = (pos_x*1.0)/box_size_x-0.5-0.5/box_size_x;
    float eta = (pos_y*1.0)/box_size_y-0.5-0.5/box_size_y;
    float weighting = 9*(4*pow(zeta,2)-4*fabs(zeta)+1) * (4*pow(eta,2)-4*fabs(eta)+1);

    subImg[i].x = subImg[i].x * weighting;
}
