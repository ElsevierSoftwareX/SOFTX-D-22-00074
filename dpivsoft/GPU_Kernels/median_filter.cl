void sort(float *a, float *b, float *c){
    float swap;
    if(*a > *b){
        swap = *a;
        *a = *b;
        *b = swap;
    }
    if(*a > *c){
        swap = *a;
        *a = *c;
        *c = swap;
    }
    if(*b > *c){
        swap = *b;
        *b = *c;
        *c = swap;
    }
}

KERNEL void Median_Filter(GLOBAL_MEM float *uf,
GLOBAL_MEM float *vf,
GLOBAL_MEM float *u,
GLOBAL_MEM float *v,
GLOBAL_MEM float *median_limit,
GLOBAL_MEM int *data)
{
    float u_median, v_median;
    float limit = *median_limit;

    int width = data[0];
    int no_boxes_x = data[4];
    int no_boxes_y = data[5];

    const SIZE_T i = get_global_id(0);

    int pos_y = i/no_boxes_x;
    int pos_x = i%no_boxes_x;

    float u00=0, u01=0, u02=0, u10=0, u11=0, u12=0, u20=0, u21=0, u22=0;
    float v00=0, v01=0, v02=0, v10=0, v11=0, v12=0, v20=0, v21=0, v22=0;

    float outdomain = -INFINITY;

    u11 = u[i];
    v11 = v[i];

    if (pos_x == 0){
        u00 = outdomain;
        u10 = outdomain;
        u20 = outdomain;

        v00 = outdomain;
        v10 = outdomain;
        v20 = outdomain;
    } else{
        u10 = u[i-1];
        v10 = v[i-1];
    }
    if (pos_x == no_boxes_x-1){
        u02 = outdomain;
        u12 = outdomain;
        u22 = outdomain;

        v02 = outdomain;
        v12 = outdomain;
        v22 = outdomain;
    }else{
        u12 = u[i+1];
        v12 = v[i+1];
    }
    if (pos_y == 0){
        u00 = outdomain;
        u01 = outdomain;
        u02 = outdomain;

        v00 = outdomain;
        v01 = outdomain;
        v02 = outdomain;
    }else{
        u01 = u[i-no_boxes_x];
        v01 = v[i-no_boxes_x];
    }
    if (pos_y == no_boxes_y-1){
        u20 = outdomain;
        u21 = outdomain;
        u22 = outdomain;

        v20 = outdomain;
        v21 = outdomain;
        v22 = outdomain;
    }else{
        u21 = u[i+no_boxes_x];
        v21 = v[i+no_boxes_x];
    }
    if(u00 != outdomain)
    {
        u00 = u[i-no_boxes_x-1];
        v00 = v[i-no_boxes_x-1];
    }
    if(u02 != outdomain)
    {
        u02 = u[i-no_boxes_x+1];
        v02 = v[i-no_boxes_x+1];
    }
    if(u20 != outdomain)
    {
        u20 = u[i+no_boxes_x-1];
        v20 = v[i+no_boxes_x-1];
    }
    if(u22 != outdomain)
    {
        u22 = u[i+no_boxes_x+1];
        v22 = v[i+no_boxes_x+1];
    }



    //Sort u_rows
    sort(&(u00), &(u01), &(u02));
    sort(&(u10), &(u11), &(u12));
    sort(&(u20), &(u21), &(u22));

    //sort u_columns
    sort(&(u00), &(u10), &(u20));
    sort(&(u01), &(u11), &(u21));
    sort(&(u02), &(u12), &(u22));

    //Sort v_rows
    sort(&(v00), &(v01), &(v02));
    sort(&(v10), &(v11), &(v12));
    sort(&(v20), &(v21), &(v22));

    //sort v_columns
    sort(&(v00), &(v10), &(v20));
    sort(&(v01), &(v11), &(v21));
    sort(&(v02), &(v12), &(v22));


    if (u01 == outdomain && u10 == outdomain){
        u_median = (u12+u21)/2;
        v_median = (v12+v21)/2;
    }else if(u01 == outdomain){
        sort(&(u00), &(u12), &(u21));
        sort(&(u00), &(u11), &(u20));
        sort(&(v00), &(v12), &(v21));
        sort(&(v00), &(v11), &(v20));
        u_median = (u20+u12)/2;
        v_median = (v20+v12)/2;
    }else if(u10 == outdomain){
        sort(&(u00), &(u02), &(u11));
        sort(&(u00), &(u12), &(u21));
        sort(&(v00), &(v02), &(v21));
        sort(&(v00), &(v12), &(v21));
        u_median = (u11+u12)/2;
        v_median = (v11+v12)/2;
    }else{
        //sort the diagonal
        sort(&(v00), &(v11), &(v22));
        sort(&(u00), &(u11), &(u22));
        u_median = u11;
        v_median = v11;
    }

    float median_magnitude = sqrt(u_median*u_median+v_median*v_median);
    float delta_u = sqrt(pow(u[i]-u_median,2)+pow(v[i]-v_median,2));


    if (isnan(u_median) || isnan(v_median)){
        uf[i] = 0;
        vf[i] = 0;
    }
    else if (delta_u > median_magnitude*limit){
        uf[i] = u_median;
        vf[i] = v_median;
    }
    else{
        uf[i] = u[i];
        vf[i] = v[i];
    }
}
