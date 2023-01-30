//Normalize and center in 0 the subimages

KERNEL void Normalize(GLOBAL_MEM float2 *subImg,
GLOBAL_MEM float *gamma,
GLOBAL_MEM int *data)
{
    int box_size_x = data[2];
    int box_size_y = data[3];

    const SIZE_T i = get_global_id(0);
    int pos = i/(box_size_x*box_size_y);

    // Act only on pixels outside the mask so the mask
    // becomes the mean intensity value
    if (subImg[i].x)
    {
        //substract the mean value
        subImg[i].x = subImg[i].x - gamma[pos];
    }
 }
