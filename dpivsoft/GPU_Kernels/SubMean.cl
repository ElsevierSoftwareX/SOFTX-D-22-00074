//Obtain mean value of each sub-image box

KERNEL void SubMean(GLOBAL_MEM float *gamma,
GLOBAL_MEM float2 *subImg,
GLOBAL_MEM int *data)
{
    int box_size_x = data[2];
    int box_size_y = data[3];
    float temp = 0;
    int idx;
    int cont = 0;

    const SIZE_T pos = get_global_id(0);

    for (int j=0; j<box_size_y; j++)
    {
        for (int i=0; i<box_size_x; i++)
        {
            idx = pos*box_size_y*box_size_x+box_size_x*j+i;
            //Add only not masked values
            if (subImg[idx].x){
                temp = temp+subImg[idx].x;
                cont = cont+1;
            }
        }
    }
    //mean of not masked values
    gamma[pos] = temp/cont;
}
