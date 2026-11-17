#include "clutil.h"
    
   /* functions used to compute the ppf values */
    
   float4 compute_ppf(float4 p1, float4 n1, float4 p2, float4 n2)
   {
        float4 d = p2 - p1;

        float4 f;
        f.x = length(d);

        float4 norm_n1 = normalize(n1);
        float4 norm_n2 = normalize(n2);
        float4 norm_d = normalize(d);

        f.y = acos(dot(norm_d, norm_n1));
        f.z = acos(dot(norm_d, norm_n2));
        f.w = acos(dot(norm_n1, norm_n2));
        
        return f;
   }
 

 
   /* discretize the given feature key */
   float4 disc_feature(float4 f, float d_dist, float d_angle)
   {
       float4 out;
       out.x = quant_downf(f.x, d_dist);
       out.y = quant_downf(f.y, d_angle);
       out.z = quant_downf(f.z, d_angle);
       out.w = quant_downf(f.w, d_angle);
       return out;
   }
   
 
   __kernel void ppfKernel(__global float4 *points, __global float4 *norms, __global float4 *out, int count,
                           uint ref_point_downsample_factor, float d_dist, int isScene , float sceneRaduis)
   {
   
       if(count <= 1) return;

        int idx = get_global_id(0);
        int bound;
        
          while(idx < count) 
          {
   
                    float4 thisPoint = points[idx];
                    float4 thisNorm  = norms[idx];
            
                    for(int i = 0; i < count; i+=BLOCK_SIZE)
                    {
            
                        bound = min(count - i, BLOCK_SIZE);
                        //barrier(CLK_GLOBAL_MEM_FENCE);
            
                        for(int j = 0; j < bound; j++) {
            
                            if(idx % ref_point_downsample_factor != 0)
                            {
                                out[idx*count + j + i].x = NAN;
                                continue;
                            }
            
                            /* handle case of identical points in pair */
                            if((j + i - idx) == 0)
                            {
                                out[idx*count + j + i].x = NAN;
                                continue;
                            }

                            /* [Vidal] Pair points that are close enough to the current reference point (distance less than the model diameter divided by 2) */
                                if(isScene == 1)
                                {
                                    float4 d = thisPoint - points[i + j];

                                    float dist_pair_threshold = length(d);

                                    if(dist_pair_threshold >= sceneRaduis)
                                    {
                                        out[idx*count + j + i].x = NAN;
                                        continue;
                                    }
                        
                                }

                            float4 ppf = compute_ppf(thisPoint, thisNorm, points[i + j], norms[i + j]);
                            out[idx*count + j + i] = disc_feature(ppf, d_dist, (float)D_ANGLE0);
                        }
                    }
                        //grid stride
                       // barrier(CLK_GLOBAL_MEM_FENCE);
                        idx += get_global_size(0);
             }
   }
