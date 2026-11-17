#include "clutil.h"


__kernel void rotClusteringKernel(    __global float4 *translations,
                                      __global float4 *quaternions,
                                      __global float *vote_counts,
                                      __global uint *adjacent_trans_hash,
                                      __global size_t *transIndices,
                                      __global uint *transKeys,  
                                      __global size_t *transCount,
                                      __global size_t *firstTransIndex,
                                      __global size_t *key2transMap,
                                      __global float *vote_counts_out,
                                      int count, float trans_thresh,
                                      uint use_l1_norm, uint use_averaged_clusters)
{
    if(count <= 1) return;

    // the origina implementation this is equal to 2 * d_angles
    float ROT_THRESH = 0.5f * (float)D_ANGLE0;

    int idx = get_global_id(0);

    float rot_thresh_sq = ROT_THRESH*ROT_THRESH;

    while(idx < count)
    {
        float4 thisTrans = translations[idx];
        float4 thisQuat  = quaternions[idx];
        float vote_count_out = 1.0f;
        float4 thisTrans_out = thisTrans;
        for(int adjacent_block = 0; adjacent_block < 27; adjacent_block++)
        {
            uint thisAdjacentHash = adjacent_trans_hash[27*idx+adjacent_block];
            uint thisAdjacentHashIndex = transIndices[27*idx+adjacent_block];
            if(thisAdjacentHash == 0 || thisAdjacentHash != transKeys[thisAdjacentHashIndex])
            {
                continue;
            }

            uint thisTransCount = transCount[thisAdjacentHashIndex];
            uint thisFirstTransIndex = firstTransIndex[thisAdjacentHashIndex];

            for(int j = 0; j < thisTransCount; j++)
            {
                uint  thisTransIndex = key2transMap[thisFirstTransIndex+j];
                float thisVoteCount = vote_counts[thisTransIndex];
                float quatDiff = fabs(8.0f * (1.0f - dot(thisQuat, quaternions[thisTransIndex])));

                if(quatDiff < rot_thresh_sq)
                {
                    if(!(use_l1_norm == 1))
                    {
                        float normDiffTrans =
                            length(thisTrans - translations[thisTransIndex]);
                        if(!(normDiffTrans < trans_thresh)){
                            continue;
                        }
                    }
                    if(use_averaged_clusters == 1)
                    {
                        thisTrans_out = vote_count_out * thisTrans_out;
                        thisTrans_out = thisTrans_out + vote_counts[thisTransIndex] * translations[thisTransIndex];
                        thisTrans_out = (1.0f / (vote_count_out + thisVoteCount)) * thisTrans_out;
                    }
                    vote_count_out += thisVoteCount;
                }
            }
        }
        vote_counts_out[idx] = vote_count_out;
        translations[idx] = thisTrans_out;

        // grid stride
        idx += get_global_size(0);
    }
}
