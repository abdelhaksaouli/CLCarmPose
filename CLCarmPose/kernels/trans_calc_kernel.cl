
#include "clutil.h" 


// my be put this function inside its own headers
uint low_32(ulong i)
{
    return (uint) (i & (-1ul >> 32));
}

uint high_32(ulong i)
{
    return (uint) (i >> 32);
}


// coordinate system transformation
 float4 compute_rot_angles(float4 n_r_m, float4 n_r_s)
 {
    mat4 rot_y;
    float4 n_tmp;
    float4 rot_angles = (float4)(0.0f,0.0f,0.0f,0.0f);

    float m_roty = atan2(n_r_m.z, n_r_m.x);
    rot_y = rotY(m_roty);

    n_tmp = n_r_m;
    n_tmp.w = 1.0f;

    n_tmp = mat4_mul_v4(rot_y, n_tmp);

    float m_rotz = -1.0f * atan2(n_tmp.y, n_tmp.x);

    float s_roty = atan2(n_r_s.z, n_r_s.x);
    rot_y = rotY(s_roty);

    n_tmp = n_r_s;
    n_tmp.w = 1.0f;
   
    n_tmp = mat4_mul_v4(rot_y, n_tmp);
    
    float s_rotz = -1.0f * atan2(n_tmp.y, n_tmp.x);

    rot_angles.x = m_roty;
    rot_angles.y = m_rotz;

    rot_angles.z = s_roty;
    rot_angles.w = s_rotz;

    return rot_angles;
}

 mat4 compute_transforms( uint angle_idx, float4 m_r, float m_roty, float m_rotz,
                                          float4 s_r, float s_roty, float s_rotz)
{
    mat4 transm, rot_x, rot_y, rot_z, T_tmp, T_m_g, T_s_g, T, T_tmp2;
  
    m_r = -1.0f * m_r;
    transm = trans(m_r);


    rot_y = rotY(m_roty);
    rot_z = rotZ(m_rotz);

    T_tmp = multiply4x4(rot_z, rot_y);
    T_m_g = multiply4x4(T_tmp, transm);

    s_r = -1.0f * s_r;
    transm = trans(s_r);

    rot_y = rotY(s_roty);
    rot_z = rotZ(s_rotz);

    T_tmp = multiply4x4(rot_z, rot_y);
    T_s_g = multiply4x4(T_tmp, transm);

    rot_x = rotX(angle_idx * (float)D_ANGLE0 - M_PI_F);

    T_tmp = invert4x4(T_s_g);

    T_tmp2 = multiply4x4(T_tmp, rot_x);

    T      = multiply4x4(T_tmp2, T_m_g);

    return T;
}

__kernel void transCalcKernel(__global ulong *votes, __global float4 *model_points, __global float4 *model_normals,
                                   __global float4 *scene_points, __global float4 *scene_normals,
                                   __global float *transforms, int count)
    {
    if(count <= 1) return;

    int idx = get_global_id(0);

     uint angle_idx, model_point_idx, scene_point_idx, modelanglecode;
     ulong vote;

    uint low6 = ((uint) -1) >> 26;


    float4 rot_angles;

    while(idx < count)
    {
        vote = votes[idx];
        scene_point_idx = high_32(vote);
        modelanglecode  = low_32(vote);

        model_point_idx = (modelanglecode >> 6);
        angle_idx = modelanglecode & low6;

        if(scene_point_idx == 0 && model_point_idx == 0 && angle_idx == 0)
        {
            idx += get_global_size(0);
            continue;
        }

        rot_angles = compute_rot_angles(model_normals[model_point_idx], scene_normals[scene_point_idx]);

        mat4 transofrm_indx = compute_transforms(angle_idx, model_points[model_point_idx],
                           rot_angles.x, rot_angles.y,
                           scene_points[scene_point_idx],
                           rot_angles.z, rot_angles.w);
         
         // do this the brutal way i dont want to create structure mat4 for the host and shit
          transforms[0 + idx*16] = transofrm_indx.matrix[0][0];
          transforms[1 + idx*16] = transofrm_indx.matrix[0][1];
          transforms[2 + idx*16] = transofrm_indx.matrix[0][2];
          transforms[3 + idx*16] = transofrm_indx.matrix[0][3];

          transforms[4 + idx*16] = transofrm_indx.matrix[1][0];
          transforms[5 + idx*16] = transofrm_indx.matrix[1][1];
          transforms[6 + idx*16] = transofrm_indx.matrix[1][2];
          transforms[7 + idx*16] = transofrm_indx.matrix[1][3];

          transforms[8 + idx*16]  = transofrm_indx.matrix[2][0];
          transforms[9 + idx*16]  = transofrm_indx.matrix[2][1];
          transforms[10 + idx*16] = transofrm_indx.matrix[2][2];
          transforms[11 + idx*16] = transofrm_indx.matrix[2][3];
         
          transforms[12 + idx*16] = transofrm_indx.matrix[3][0];
          transforms[13 + idx*16] = transofrm_indx.matrix[3][1];
          transforms[14 + idx*16] = transofrm_indx.matrix[3][2];
          transforms[15 + idx*16] = transofrm_indx.matrix[3][3];

        //grid stride
        idx += get_global_size(0);
    }
}