/*!************************************************************
* \copyright (C) Copyright LaTIM\n
*            This software is the confidential and proprietary
*            information of LaTIM. You shall not disclose such
*            confidential information and shall use it only in
*            accordance with the terms of the license agreement
*            you entered into with LaTIM.
*
*	     Author: Abdelhak SAOULI
***************************************************************/

// PCL headers
#include <Eigen/Core>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

//Opencl mangers
#include "defines.h"
#include "scene.h"
#include "clParallelHashArray.h"

//C/C++ headers
#include <iostream>
#include <vector>

#pragma once
class Model : public Scene
{
	public:
		Model(pcl::PointCloud<pcl::PointNormal>* cloud, 
			std::vector<float_> modelWights/* jerom 2020 -- number of point in each voxel used as wieght*/,
			float_ d_dist, 
			float_ vote_count_threshold,
			uint_ use_l1_norm, uint_ use_averaged_clusters);

		~Model();

        
        void ComputeUniqueVotes(Scene* scene);
        boostCL::vector<float_>* ComputeWeightedVoteCounts(boostCL::vector<ulong_>& votes,
                                                          boostCL::vector<uint_>& vote_counts,
                                                          boostCL::vector<float_>& modelpoint_vote_weights);


        void ComputeTransformations(Scene* scene);
        boostCL::vector<float_>* ClusterTransformations();
        void PPFEstimate(Scene* scene);

        boostCL::vector<float_>* getTransformations()
        {
            return transformations;
        };

        Eigen::Matrix4f getPose()const;

    public:
        float_ vote_count_threshold;
        uint_ use_l1_norm;
        uint_ use_averaged_clusters;

        // ppfCount[i] is the number of PPFs whose hash is hashKeys[i];
        boostCL::vector<uint_>* ppfCount;

        // Vector of all vote codes [Scene Ref Point|Model Point|Angle]
        //                           63           32 31        6 5   0
        boostCL::vector<ulong_>* votes;

        boostCL::vector<uint_>* voteCounts;

        boostCL::vector<float_>* transformations;

        boostCL::vector<float4_>* transformation_rots;
        boostCL::vector<float4_>* transformation_trans;
        boostCL::vector<float_>* vote_counts_out;


        clParallelHashArray<uint_> search_array;

        boostCL::vector<float_>* modelPointVoteWeights;
        boostCL::vector<float_>* weightedVoteCounts;
        uint_ max_idx;
};

