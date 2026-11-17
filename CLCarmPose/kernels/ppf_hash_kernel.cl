/* https://stackoverflow.com/questions/24420246/c-function-to-convert-float-to-byte-array */

uint hash( float4 f, uint hash)
{
   union 
   {
        float4 a;
        char bytes[sizeof(float4)];
    } data;

    data.a = f;

    for(int i = 0; i < sizeof(float4); i++)
    {
        hash *= 16777619;
        hash ^= data.bytes[i];
    }
    return hash;
}
   
 __kernel void ppfHashKernel(__global float4 *ppfs, __global uint *codes, int count)
{
       if(count <= 1) return;
  
       int idx = get_global_id(0);
   
       while(idx < count)
       {
           if(isnan(ppfs[idx].x))
           {
               codes[idx] = 0;
           }
           else
           {
               codes[idx] = hash(ppfs[idx], 2166136261);
           }
   
           //grid stride
           idx += get_global_size(0);
       }
}