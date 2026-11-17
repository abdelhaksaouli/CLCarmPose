#include "clutil.h"

float4 translationFromMat4(mat4 t)
{
    // i set the w part of the out put to 1 we actually never use it in calculations
    return (float4)(t.matrix[0][3],t.matrix[1][3],t.matrix[2][3], 1.0);
}

__kernel void mat2transquatKernel(__global float *transformations, __global float4 *transformation_trans, __global float4 *transformation_rots, int count)

{
    if(count <= 1) return;

    int idx = get_global_id(0);

    while(idx < count)
    {
        mat4 transform;
        // do deep copy which  is bad and cost in performance i guess

          transform.matrix[0][0] = transformations[0 + idx*16];
          transform.matrix[0][1] = transformations[1 + idx*16];
          transform.matrix[0][2] = transformations[2 + idx*16];
          transform.matrix[0][3] = transformations[3 + idx*16];

          transform.matrix[1][0] = transformations[4 + idx*16];
          transform.matrix[1][1] = transformations[5 + idx*16];
          transform.matrix[1][2] = transformations[6 + idx*16];
          transform.matrix[1][3] = transformations[7 + idx*16];

          transform.matrix[2][0] = transformations[8 + idx*16];
          transform.matrix[2][1] = transformations[9 + idx*16];
          transform.matrix[2][2] = transformations[10 + idx*16];
          transform.matrix[2][3] = transformations[11 + idx*16];
         
          transform.matrix[3][0] = transformations[12 + idx*16];
          transform.matrix[3][1] = transformations[13 + idx*16];
          transform.matrix[3][2] = transformations[14 + idx*16];
          transform.matrix[3][3] = transformations[15 + idx*16];

          transformation_trans[idx] = translationFromMat4(transform);
          transformation_rots[idx]  = hrotmat2quat(transform);
        
        idx += get_global_size(0);
    }
}

