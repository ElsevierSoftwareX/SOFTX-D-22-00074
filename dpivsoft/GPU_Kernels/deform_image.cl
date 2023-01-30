// Deform image

KERNEL void Deform_image(GLOBAL_MEM float2 *subImg1,
GLOBAL_MEM float2 *subImg2,
GLOBAL_MEM float *img1,
GLOBAL_MEM float *img2,
GLOBAL_MEM int *box_origin_x,
GLOBAL_MEM int *box_origin_y,
GLOBAL_MEM float *u,
GLOBAL_MEM float *v,
GLOBAL_MEM float *du_dx,
GLOBAL_MEM float *du_dy,
GLOBAL_MEM float *dv_dx,
GLOBAL_MEM float *dv_dy,
GLOBAL_MEM float *u_index,
GLOBAL_MEM float *v_index,
GLOBAL_MEM int *data)
{
    float temp;

    int width = data[0];
    int height = data[1];
    int box_size_x = data[2];
    int box_size_y = data[3];
    int no_boxes_x = data[4];
    int no_boxes_y = data[5];

    const SIZE_T pos = get_global_id(0);
    int pos1 = pos/(box_size_y*box_size_x);
    int pos2 = pos%(box_size_y*box_size_x);

    int i_matrix = pos2 % box_size_x;
    int j_matrix = pos2 / box_size_x;

    int i_index = box_origin_x[pos1] + i_matrix;
    int j_index = box_origin_y[pos1] + j_matrix;

    u_index[pos] = u[pos1] + du_dx[pos1]*(i_matrix-box_size_x/2) +
        du_dy[pos1]*(j_matrix-box_size_y/2);
    v_index[pos] = v[pos1] + dv_dx[pos1]*(i_matrix-box_size_x/2) +
        dv_dy[pos1]*(j_matrix-box_size_y/2);

    temp = (i_index - u_index[pos]/2);
    float i_frac_1 = temp-(int)temp;
    int i_index_1 = (int)temp;

    temp = (i_index + u_index[pos]/2);
    float i_frac_2 = temp-(int)temp;
    int i_index_2 = (int)temp;

    temp =(j_index - v_index[pos]/2);
    float j_frac_1 = temp-(int)temp;
    int j_index_1 = (int)temp;

    temp = (j_index + v_index[pos]/2);
    float j_frac_2 = temp-(int)temp;
    int j_index_2 = (int)temp;

    if (i_index_1 <= 3 || i_index_1 > width-2 || i_index_2 <= 3 || i_index_2 > width-2) {
        i_index_1 = i_index;
        i_index_2 = i_index;
        u_index[pos] = 0;
    }
    if (j_index_1 <= 0 || j_index_1 > height-2 || j_index_2 <= 3 || j_index_2 > height-2){
        j_index_1 = j_index;
        j_index_2 = j_index;
        v_index[pos] = 0;
    }


    subImg1[pos].x = (1-j_frac_1)*(1-i_frac_1)*img1[j_index_1*width+i_index_1]
        +(1-j_frac_1)*(i_frac_1)*img1[(j_index_1)*width+i_index_1+1]
        +(j_frac_1)*(1-i_frac_1)*img1[(j_index_1+1)*width+i_index_1]
        +(j_frac_1)*(i_frac_1)*img1[(j_index_1+1)*width+i_index_1+1];

    subImg2[pos].x = (1-j_frac_2)*(1-i_frac_2)*img2[j_index_2*width+i_index_2]
        +(1-j_frac_2)*(i_frac_2)*img2[(j_index_2)*width+i_index_2+1]
        +(j_frac_2)*(1-i_frac_2)*img2[(j_index_2+1)*width+i_index_2]
        +(j_frac_2)*(i_frac_2)*img2[(j_index_2+1)*width+i_index_2+1];

    subImg1[pos].y = 0;
    subImg2[pos].y = 0;
}

