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


#pragma once

#include "PCLoader.h"
#include "PCTools.h"
#include "clContextManger.h"

#include "scene.h"
#include "model.h"

//eigen headers
#include <Eigen/Core>

class GPUPPFEstimator
{
	public : 
		GPUPPFEstimator();
		~GPUPPFEstimator();

	public:
		void init_device();

    private:
		void enable_cuda_build_cache(bool enable);

    private:
        // some algorithm settings
        std::string model_files;
        std::string scene_files;

        float_ scene_leaf_size;
        float_ tau_d;
        float_ downsamplingAngle; // angle to use as threshold for downsampling

        uint_ ref_point_df;


        pcl::PointNormal picked_point;
        Model* cad;	/* the downsampled model */

        Scene* scn;	/* the downsampled scene */


        float_ model_d_dists;                   /* model discritization distance used in downsampling and in ppf discritization 
                                                   its equal to t_au_d * model diam */
        float_ scene_d_dists;						/* scene discritization distance used only in downsampling  */


           /* sceneSearchRaduis is the raduis allowed for point to be paired in,
            * if distance betewen 2 two point in the scene bigger then this raduis
            * then don't consider pairing them togther
            */

        float_ sceneSearchRaduis;

        /* vector that will contain the down sampling rate which will be used as wieght for voting Jerom (2020) */
        std::vector<float_> modelWight;

        /* the threshold for voting, this should not be a parameter because it is computed based on wieghts */
        float_ vote_count_threshold;

        /* the end game result */
        Eigen::Matrix4f  results;
     private:
        /* compute  vote_count_threshold based on sampling rate [Jerom 2020] */
        void computeVoteCountThreshold();
        void readParameters();  // read the application parameters from Text file

    public:
        void trainModel();
        void setUpScene();
        void PPFGPURegistration();

       
};

inline void GPUPPFEstimator::enable_cuda_build_cache(bool enable)
       {
            #ifdef _MSC_VER
                    if (enable)
                        _putenv("CUDA_CACHE_DISABLE=0");
                    else
                        _putenv("CUDA_CACHE_DISABLE=1");
            #else // GCC
                    if (enable)
                        putenv("CUDA_CACHE_DISABLE=0");
                    else
                        putenv("CUDA_CACHE_DISABLE=1");
            #endif
        }