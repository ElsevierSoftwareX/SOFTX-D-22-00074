//Find Peak Kernel
KERNEL void directCorrelation(
        GLOBAL_MEM float *Img1,
        GLOBAL_MEM float *Img2,
        GLOBAL_MEM float *correlation,
        GLOBAL_MEM int *box_origin_x,
        GLOBAL_MEM int *box_origin_y,
        GLOBAL_MEM int *data)
{
    const SIZE_T i = get_global_id(0);

    int width = data[0];
    int box_size_x = data[2];
    int box_size_y = data[3];
    int window_x = data[6];
    int window_y = data[7];
    int weighting = 1;


    int box_size_x_d = round(window_x/2.0)*2+1;
    int box_size_y_d = round(window_y/2.0)*2+1;

    int pos = i/(box_size_y_d*box_size_x_d);

    // Obtain the pixels movement of the direct correlation
    int ii = (i%(box_size_y_d*box_size_x_d))%box_size_x_d - round(window_y/2.0);
    int jj = i%(box_size_y_d*box_size_x_d)/box_size_x_d - round(window_x/2.0);
    int k  = (box_origin_y[pos]+jj)*width+box_origin_x[pos]+ii;

    int box_o_y = box_origin_y[pos]+round(jj/2.0);
    int box_o_x = box_origin_x[pos]+round(ii/2.0);

    float subImg1, subImg2;
    float temp = 0, temp1 = 0, temp2 = 0;
    float zeta, eta, weight_factor;

    for (int j=0; j<box_size_y; j++)
    {
        for (int x=0; x<box_size_x; x++)
        {
            subImg1 = 1.0*Img1[(box_o_y-jj+j)*width+box_o_x-ii+x];
            subImg2 = 1.0*Img2[(box_o_y+j)*width+box_o_x+x];

            if (weighting)
            {
                zeta = (x*1.0)/(box_size_x)-0.5-0.5/(box_size_x);
                eta = (j*1.0)/(box_size_y)-0.5-0.5/(box_size_y);
                weight_factor = 9*(4*pow(zeta,2)-4*fabs(zeta)+1) * (4*pow(eta,2)-4*fabs(eta)+1);
                subImg1 = subImg1*weight_factor;
                subImg2 = subImg2*weight_factor;
            }

            temp1 += subImg1;
            temp2 += subImg2;

            temp += subImg1*subImg2;
        }
    }

    // Scale image
    temp1 = temp1/(box_size_y*box_size_x);
    temp2 = temp2/(box_size_y*box_size_x);

    //int Sigma1 = max(0.1, sqrt(sum(pow(SubImg1, 2))));
    //int Sigma2 = max(0.1, sqrt(sum(pow(SubImg2, 2))));

    correlation[i] = temp-(temp1+temp2);

}
