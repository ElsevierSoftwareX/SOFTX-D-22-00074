//Find Peak Kernel
KERNEL void find_peak_direct(GLOBAL_MEM float *v,
GLOBAL_MEM float *u,
GLOBAL_MEM float *input,
GLOBAL_MEM int *data)
{
    int window_x = data[6];
    int window_y = data[7];
    int peak = data[8];
    float peak_noise = peak/1000.0;

    int box_size_x = round(window_x/2.0)*2+1;
    int box_size_y = round(window_y/2.0)*2+1;


    int idx, idrow, idcol;
    int idx1, idrow1 = 0, idcol1 = 0;
    int idx2, idrow2 = 0, idcol2 = 0;
    int lm = 4;
    float temp1 = 0, temp2 = 0;
    float max_peak1 = 0, max_peak2 = 0;
    float epsilon_x = 0, epsilon_y = 0;
    float fit_peak_00, fit_peak_01, fit_peak_02, fit_peak_10, fit_peak_11;
    float fit_peak_12, fit_peak_20, fit_peak_21, fit_peak_22;


    const SIZE_T pos = get_global_id(0);


    //Find first peak
    for (int j=4; j<box_size_y-3; j++)
    {
        for (int i=4; i<box_size_x-3; i++)
        {
            idx1 = pos*box_size_y*box_size_x+box_size_x*j+i;
            if(input[idx1]>max_peak1)
            {
                max_peak1 = input[idx1];
                idrow1 = j;
                idcol1 = i;
            }
        }
    }

    idx1 = pos*box_size_y*box_size_x+box_size_x*idrow1+idcol1;
    temp1 = (input[idx1+1]+input[idx1-1]+input[idx1+box_size_x] +
           input[idx1+box_size_x+1]+input[idx1+box_size_x-1] +
           input[idx1-box_size_x]+ input[idx1-box_size_x+1] +
           input[idx1-box_size_x-1]);

    //Find second peak
    for (int j=4; j<box_size_y-3; j++)
    {
        for (int i=4; i<box_size_x-3; i++)
        {
            idx2 = pos*box_size_y*box_size_x+box_size_x*j+i;
            if(input[idx2]>max_peak2 && abs(i-idcol1)>lm && abs(j-idrow1)>lm )
            {
                max_peak2 = input[idx2];
                idrow2 = j;
                idcol2 = i;
            }
        }
    }

    idx2 = pos*box_size_y*box_size_x+box_size_x*idrow2+idcol2;
    temp2 = (input[idx2+1]+input[idx2-1]+input[idx2+box_size_x]+
              input[idx2+box_size_x+1]+input[idx2+box_size_x-1]+
              input[idx2-box_size_x]+input[idx2-box_size_x+1]+
              input[idx2-box_size_x-1]);

    if (temp1 > temp2)
    {
        idx = idx1;
        idrow = idrow1;
        idcol = idcol1;
    }
    else
    {
        idx = idx2;
        idrow = idrow2;
        idcol = idcol2;
    }


    float box_size_f_x = box_size_x;
    float box_size_f_y = box_size_y;
    float lim = 0.001;

    //Check noise to peak ratio
    if (max_peak2/(max_peak1+1e-10) > peak_noise)
    {
        fit_peak_00 = 0;
        fit_peak_01 = 0;
        fit_peak_02 = 0;
        fit_peak_10 = 0;
        fit_peak_11 = 0;
        fit_peak_12 = 0;
        fit_peak_20 = 0;
        fit_peak_21 = 0;
        fit_peak_22 = 0;
    }
    else
    {
        fit_peak_00 = input[idx-box_size_x-1];
        fit_peak_01 = input[idx-box_size_x];
        fit_peak_02 = input[idx-box_size_x+1];
        fit_peak_10 = input[idx-1];
        fit_peak_11 = input[idx];
        fit_peak_12 = input[idx+1];
        fit_peak_20 = input[idx+box_size_x-1];
        fit_peak_21 = input[idx+box_size_x];
        fit_peak_22 = input[idx+box_size_x+1];
    }


    //Sub-pixel accuracy with gaussian estimator
    if (fit_peak_11 <= fit_peak_10 && fit_peak_11 <= fit_peak_12)
    {
        epsilon_x = 0;
        idcol = box_size_x/2;
    }
    else
    {
        if (fit_peak_10 > 0 && fit_peak_11 > 0 && fit_peak_12 > 0 &&
                       (log(fit_peak_10)+log(fit_peak_12)-2*log(fit_peak_11)) != 0)
        {
            epsilon_x = (0.5*(log(fit_peak_10)-log(fit_peak_12))/(log(fit_peak_10)+
                log(fit_peak_12)-2*log(fit_peak_11)));
        }
        else
        {
            epsilon_x = 0;
        }
    }

    if (fit_peak_11 <= fit_peak_01 && fit_peak_11 <= fit_peak_21)
    {
        epsilon_y = 0;
        idrow = box_size_y/2;
    }
    else
    {
        if (fit_peak_01 > 0 && fit_peak_11 > 0 &&  fit_peak_21 > 0 &&
                        (log(fit_peak_01)+log(fit_peak_21)-2*log(fit_peak_11)) != 0)
        {
            epsilon_y = (0.5*(log(fit_peak_01)-log(fit_peak_21))/(log(fit_peak_01)+
                        log(fit_peak_21)-2*log(fit_peak_11)));
        }
        else
        {
            epsilon_y = 0;
        }
    }

    u[pos] = idcol + epsilon_x - box_size_x/2;
    v[pos] = idrow + epsilon_y - box_size_y/2;
}
