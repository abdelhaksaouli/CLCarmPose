
uint low_32(ulong i)
{
    return (uint) (i & (-1ul >> 32));
}


__kernel void voteWeightKernel(__global ulong *votes, __global uint *vote_counts, __global float *modelPointVoteWeights, __global float *weightedVoteCounts, int count)
{
    if(count <= 1) return;

    int idx = get_global_id(0);

    while(idx < count)
    {
        unsigned int modelanglecode = low_32(votes[idx]);
        unsigned int model_point_idx = (modelanglecode >> 6);
        weightedVoteCounts[idx] = modelPointVoteWeights[model_point_idx] * vote_counts[idx];

        
        //grid stride
         idx += get_global_size(0);
    }
}