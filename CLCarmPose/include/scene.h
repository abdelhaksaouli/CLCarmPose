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

// PCL point cloud
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

//Opencl mangers
#include "defines.h"



//C/C++ headers
#include <iostream>
#include <vector>

#pragma once
class Scene
{
	public:
		Scene();
		Scene(pcl::PointCloud<pcl::PointNormal>* cloud_ptr, float_ d_dist, float_ sceneRaduis, uint_ ref_point_downsample_factor = 1);
		~Scene();

	public:
        uint_ numPoints() { return this->n; };
        boostCL::vector<float4_>* getModelPoints() { return this->modelPoints; };
        boostCL::vector<float4_>* getModelNormals() { return this->modelNormals; };
        boostCL::vector<float4_>* getModelPPFs() { return this->modelPPFs; };
        boostCL::vector<uint_>* getHashKeys() { return this->hashKeys; };

    protected:
        uint_ n;
        // Vector of model points
        boostCL::vector<float4_>* modelPoints;
        // Vector of model normals
        boostCL::vector<float4_>* modelNormals;
        // Vector of model point pair features
        boostCL::vector<float4_>* modelPPFs;
        // For a scene, hashKeys stores the hashes of all point pair features.
        // For a model, hashKeys is an array of all UNIQUE hashKeys. A binary search
        boostCL::vector<uint_>* hashKeys;

        float_ d_dist;

        void initPPFs(std::vector<float4_>* points, std::vector<float4_>* normals, int_ n,
            float_ d_dist, int_ isScene = 0, uint_ ref_point_downsample_factor = 1, float_ sceneRaduis = 0.0f);

};

