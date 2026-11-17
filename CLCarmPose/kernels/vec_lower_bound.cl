  
    /* No local memory startegy for this kernel */
    /* This is naive vectorized simple binary search */
    /* each thread will do binary search and return the */
    /* lower bound                                      */

   __kernel void lowerBound(__global uint *hashkey, __global uint *datahashkey, 
                            __global size_t *dataIndices, int hashkeySize, int datahashkeySize)
   {
       int idx = get_global_id(0);

       while (idx < datahashkeySize)
       {

           int left = 0;
           int right = hashkeySize - 1;

           /* answer is initialized to out of boundry */
           int ans = hashkeySize + 1;

           while (left <= right)
           {
               int mid = left + (right - left) / 2;

               if (hashkey[mid] >= datahashkey[idx])
               {
                   ans = mid;
                   right = mid - 1;
               }
               else
               {
                   left = mid + 1;
               }
           }
           //barrier(CLK_GLOBAL_MEM_FENCE);
           dataIndices[idx] = (size_t)ans;
           idx += get_global_size(0);
       }
   }