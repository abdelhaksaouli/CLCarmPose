
__kernel void ppfVoteCountKernel(__global uint *sceneKeys, __global size_t *sceneIndices, __global uint *hashKeys, __global size_t *ppfCount, __global ulong *ppf_vote_counts, int count)
{
       if(count <= 1) return;
  
       int idx = get_global_id(0);

    while(idx < count)
    {
        uint thisSceneKey = sceneKeys[idx];
        uint thisSceneIndex = sceneIndices[idx];

        if(thisSceneKey == 0 || thisSceneKey != hashKeys[thisSceneIndex])
        {
            ppf_vote_counts[idx] = 0;
        }
        else
        {
            ppf_vote_counts[idx] = ppfCount[thisSceneIndex];
        }

        idx += get_global_size(0);
    }
}