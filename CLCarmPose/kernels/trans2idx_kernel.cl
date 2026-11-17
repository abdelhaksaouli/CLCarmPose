#include "clutil.h"



float3 discrit_translation(float4 translation, float d_dist)
{
       float3 out;
       out.x = quant_downf(translation.x, d_dist);
       out.y = quant_downf(translation.y, d_dist);
       out.z = quant_downf(translation.z, d_dist);
       return out;
}


uint hash_trans( int3 trans, uint hash)
{
   union 
   {
        int3 a;
        char bytes[sizeof(int3)]; // we has only the
    } data;

    data.a = trans;

    for(int i = 0; i < sizeof(int3); i++)
    {
        hash *= 16777619;
        hash ^= data.bytes[i];
    }
    return hash;
}



__kernel void trans2idxKernel(__global float4 *translations, __global uint *trans_hash, __global uint *adjacent_trans_hash, int count, float d_dist)
{
    if(count <= 1) return;

    int idx = get_global_id(0);
    
    int3 norm_disc_trans;
    int3 adjacent_norm_disc_trans;

    while(idx < count)
    {
        float3 disc_trans = discrit_translation(translations[idx], d_dist);
        norm_disc_trans.x = (int) (disc_trans.x / d_dist);
        norm_disc_trans.y = (int) (disc_trans.y / d_dist);
        norm_disc_trans.z = (int) (disc_trans.z / d_dist);

        //norm_disc_trans.x = (int) (disc_trans.x);
        //norm_disc_trans.y = (int) (disc_trans.y);
        //norm_disc_trans.z = (int) (disc_trans.z);

        trans_hash[idx] = hash_trans(norm_disc_trans, 2166136261);

        // Optimize this ?
        for(int i = -1, c = 0; i < 2; i++)
        {
            for(int j = -1; j < 2; j++)
            {
                for(int k = -1; k < 2; k++, c++)
                {
                    if(i == 0 && j == 0 && k == 0)
                    {
                        adjacent_trans_hash[27*idx+c] = 0;
                        continue;
                    }
                    adjacent_norm_disc_trans.x = norm_disc_trans.x + i;
                    adjacent_norm_disc_trans.y = norm_disc_trans.y + j;
                    adjacent_norm_disc_trans.z = norm_disc_trans.z + k;
                    adjacent_trans_hash[27*idx+c] = hash_trans(adjacent_norm_disc_trans,2166136261);
                }
            }
        }

        //grid stride 
        idx += get_global_size(0);
    }
}





