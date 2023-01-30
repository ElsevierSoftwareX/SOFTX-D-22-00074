// Initialize float variables to prevent wrong values at the
// first iteration due to temp_manager arrays.
//
KERNEL void ini_index(GLOBAL_MEM float *u_index,
GLOBAL_MEM float *v_index)
{
    const SIZE_T i = get_global_id(0);

    u_index[i] = 0;
    v_index[i] = 0;
}
