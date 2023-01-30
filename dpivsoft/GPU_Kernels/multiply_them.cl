//multiplication Kernel
KERNEL void multiply_them(
    GLOBAL_MEM ${ctype1} *dest,
    GLOBAL_MEM ${ctype1} *a,
    GLOBAL_MEM ${ctype2} *b)
{
  const SIZE_T i = get_global_id(0);
  dest[i] = ${mul}(a[i], b[i]);
}
