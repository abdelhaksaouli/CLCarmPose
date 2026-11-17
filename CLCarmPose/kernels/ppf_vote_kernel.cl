#include "clutil.h"




 uint trans_model_scene(float4 m_r, float4 n_r_m, float4 m_i, float4 s_r, float4 n_r_s, float4 s_i, float d_dist)
{
    mat4 transm, rot_x, rot_y, rot_z, T_tmp, T_m_g, T_s_g;
    float4 n_tmp;
    uint alpha_idx;

    m_r = -1.0f * m_r;
    transm = trans(m_r);

    rot_y = rotY(atan2(n_r_m.z, n_r_m.x));
    n_tmp = n_r_m;
    n_tmp.w = 1.0;
    n_tmp = mat4_mul_v4(rot_y, n_tmp);

    rot_z = rotZ(-1.0f * atan2(n_tmp.y, n_tmp.x));
    T_tmp = multiply4x4(rot_z, rot_y);
    T_m_g = multiply4x4(T_tmp, transm);

    s_r = -1.0f * s_r;
    transm = trans(s_r);

    rot_y = rotY(atan2(n_r_s.z, n_r_s.x));
    n_tmp = n_r_s;
    n_tmp.w = 1.0;
    n_tmp = mat4_mul_v4(rot_y, n_tmp);

    rot_z = rotZ(-1.0f * atan2(n_tmp.y, n_tmp.x));
    T_tmp = multiply4x4(rot_z, rot_y);
    T_s_g = multiply4x4(T_tmp, transm);


    n_tmp = m_i;
    n_tmp = mat4_mul_v4(T_m_g, n_tmp);
    float3 u = (float3)(n_tmp.x, n_tmp.y, n_tmp.z);

    n_tmp = s_i;
    n_tmp = mat4_mul_v4(T_s_g, n_tmp);
    float3 v = (float3)(n_tmp.x, n_tmp.y, n_tmp.z);

    u.x = 0.0f;
    v.x = 0.0f;
    float alpha = atan2(cross(u, v).x, dot(u, v));

    alpha = quant_downf(alpha + M_PI_F,(float) D_ANGLE0);
    alpha_idx = (uint) (rint((alpha/(float)D_ANGLE0)));


    return alpha_idx;
}


__kernel void ppfVoteKernel(__global uint *sceneKeys, __global size_t *sceneIndices,
                              __global uint *hashKeys, __global size_t *ppfCount,
                              __global size_t *firstPPFIndex, __global size_t *key2ppfMap,
                              __global float4 *modelPoints, __global float4 *modelNormals, int modelSize,
                              __global float4 *scenePoints, __global float4 *sceneNormals, int sceneSize,
                              __global ulong *ppf_vote_indices,
                              __global ulong *votes_old, int count, float d_dist)
{
    if(count <= 1) return;

    int idx = get_global_id(0);
    uint alpha_idx;

    while(idx < count)
    {
        uint thisSceneKey = sceneKeys[idx];
        uint thisSceneIndex = sceneIndices[idx];

        if (thisSceneKey == 0 || thisSceneKey != hashKeys[thisSceneIndex])
        {
            idx += get_global_size(0);
            continue;
        }

        uint thisPPFCount      = ppfCount[thisSceneIndex];
        uint thisFirstPPFIndex = firstPPFIndex[thisSceneIndex];

        uint scene_r_index = idx  / sceneSize;
        uint scene_i_index = idx - scene_r_index * sceneSize;

        float4 scene_r_point = scenePoints[scene_r_index];
        float4 scene_r_norm  = sceneNormals[scene_r_index];
        float4 scene_i_point = scenePoints[scene_i_index];

        uint modelPPFIndex, model_r_index, model_i_index;
        
        float4 model_r_point, model_r_norm, model_i_point;

        for(int i = 0; i < thisPPFCount; i++)
        {
            modelPPFIndex = key2ppfMap[thisFirstPPFIndex+i];
            model_r_index = modelPPFIndex / modelSize;
            model_i_index = modelPPFIndex - model_r_index*modelSize;

            model_r_point = modelPoints[model_r_index];
            model_r_norm  = modelNormals[model_r_index];
            model_i_point = modelPoints[model_i_index];
            
            alpha_idx = trans_model_scene(model_r_point, model_r_norm, model_i_point,
                            scene_r_point, scene_r_norm, scene_i_point,
                            d_dist);

            votes_old[ppf_vote_indices[idx] + i] =
                (((ulong) scene_r_index) << 32) | (model_r_index << 6) | (alpha_idx);
        }
        idx += get_global_size(0);
    }
}