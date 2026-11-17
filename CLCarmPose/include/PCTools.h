#pragma once

/*!************************************************************
 * \copyright (C) Copyright LaTIM\n
 *            This software is the confidential and proprietary
 *            information of LaTIM. You shall not disclose such
 *            confidential information and shall use it only in
 *            accordance with the terms of the license agreement
 *            you entered into with LaTIM.
 *
 *            Author: Abdelhak SAOULI
 ***************************************************************/

#include <iostream>
#include <string>
#include <algorithm>

//pcl voxel grid and stuff
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/PointIndices.h>
#include <pcl/octree/octree_pointcloud_pointvector.h>
#include <pcl/octree/octree_search.h>
#include <pcl/octree/octree_iterator.h>
#include <pcl/octree/octree_nodes.h>
#include <pcl/octree/octree.h>
#include <pcl/octree/octree_impl.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/filters/crop_box.h>

//faster normal estimation
#include <pcl/search/kdtree.h>  
#include <pcl/features/normal_3d_omp.h>

// pcl common 
#include <pcl/common/common.h>
#include <pcl/common/distances.h>

// random downsampling
#include <pcl/filters/random_sample.h>
#include <pcl/keypoints/uniform_sampling.h>


#define PCLTOOL_NAMESPACE_BEGIN namespace tools{
#define PCLTOOL_NAMESPACE_END }


PCLTOOL_NAMESPACE_BEGIN

// Random Downsampling via CPU
inline pcl::PointCloud<pcl::PointXYZ>::Ptr cpu_rand_downsample(pcl::PointCloud<pcl::PointXYZ>::Ptr pcd, float raduis)
{
    pcl::UniformSampling<pcl::PointXYZ> rs;
    typename pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    rs.setRadiusSearch(raduis);	// this shoud determing the density of the point cloud i w
    rs.setInputCloud(pcd);
    rs.filter(*filtered_cloud);
    return filtered_cloud;
}

// function that compute diameter of cloud and return max min bounding box range
inline float cloudDiam(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    pcl::PointXYZ cloud_pmin;
    pcl::PointXYZ cloud_pmax;
    pcl::getMinMax3D(*cloud, cloud_pmin, cloud_pmax);
    return pcl::euclideanDistance(cloud_pmin, cloud_pmax);
}

inline pcl::PointCloud<pcl::PointNormal>::Ptr cpu_rand_downsample(pcl::PointCloud<pcl::PointNormal>::Ptr pcd, float raduis)
{
    pcl::UniformSampling<pcl::PointNormal> rs;
    typename pcl::PointCloud<pcl::PointNormal>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointNormal>);
    rs.setRadiusSearch(raduis);	// this shoud determing the density of the point cloud i w
    rs.setInputCloud(pcd);
    rs.filter(*filtered_cloud);
    return filtered_cloud;
}

// function that compute diameter of cloud and return max min bounding box range
inline float cloudDiam(pcl::PointCloud<pcl::PointNormal>::Ptr cloud)
{
    pcl::PointNormal cloud_pmin;
    pcl::PointNormal cloud_pmax;
    pcl::getMinMax3D(*cloud, cloud_pmin, cloud_pmax);

    float x = cloud_pmax.x - cloud_pmin.x;
    float y = cloud_pmax.y - cloud_pmin.y;
    float z = cloud_pmax.z - cloud_pmin.z;
    return  std::max(x, std::max(y, z));//pcl::euclideanDistance(cloud_pmin, cloud_pmax);
}
/* un organized point cloud normal estimation */
inline pcl::PointCloud<pcl::PointNormal>::Ptr compute_normal_unorganized(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
        pcl::PointCloud<pcl::PointNormal>::Ptr output(new pcl::PointCloud<pcl::PointNormal>());
        pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
        ne.setInputCloud(cloud);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        ne.setSearchMethod(tree);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
        ne.setKSearch(30);
        //ne.setRadiusSearch(raduis);
        ne.compute(*cloud_normals);
        pcl::concatenateFields(*cloud, *cloud_normals, *output);
    return output;
}

inline pcl::PointCloud<pcl::PointNormal>::Ptr object_picking(pcl::PointCloud<pcl::PointNormal>::Ptr scene, 
                                                            pcl::PointNormal picked_point)
{
    // // declare output variable
    pcl::PointCloud<pcl::PointNormal>::Ptr cluster_organized(new pcl::PointCloud<pcl::PointNormal>);

    //https://www.siemens-healthineers.com/surgical-c-arms-and-navigation/mobile-c-arms/cios-alpha-cmos
    pcl::CropBox<pcl::PointNormal> boxFilter;
    boxFilter.setMin(Eigen::Vector4f(picked_point.x - 0.5, picked_point.y - 0.5, picked_point.z - 0.5, 1.0));
    boxFilter.setMax(Eigen::Vector4f(picked_point.x + 0.5, picked_point.y + 0.5, picked_point.z + 0.5, 1.0));
    boxFilter.setInputCloud(scene);
    boxFilter.filter(*cluster_organized);

    return cluster_organized;
}

inline pcl::PointCloud<pcl::PointNormal>::Ptr advance_voxel_grid_down_sample(pcl::PointCloud<pcl::PointNormal>::Ptr& cloud, 
                                                                      float resolution, float angle, int model ,
                                                                      std::vector<float>& modelWight)
{
    // Output pointcloud after downsampling and norma-based region growing clustering in each voxel
    pcl::PointCloud<pcl::PointNormal>::Ptr output_cloud(new pcl::PointCloud<pcl::PointNormal>);

    // Octree used for spatial partitionning with resolution resolution
    pcl::octree::OctreePointCloudPointVector<pcl::PointNormal> octree(resolution);
    octree.setInputCloud(cloud);
    octree.addPointsFromInputCloud();
    // Declaration of leafnode iterators to browse every octree leaf
    pcl::octree::OctreePointCloudPointVector<pcl::PointNormal>::LeafNodeIterator it;
    const  pcl::octree::OctreePointCloudPointVector<pcl::PointNormal>::LeafNodeIterator it_end = octree.leaf_depth_end();

    // Pointcloud representing the i-th voxel
    pcl::PointCloud<pcl::PointNormal>::Ptr voxel_cloud(new pcl::PointCloud<pcl::PointNormal>);
    // Seprarting i-th voxel pointcloud into xyz and normal pcl pointcloud to fit the region growing templated class (xyz+normal)
    pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr voxel_cloud_normals(new pcl::PointCloud<pcl::Normal>);
    // Extract indices class declaration to extract each voxel pointclouds
    pcl::ExtractIndices<pcl::PointNormal> extract_voxel_pointcloud;
    // Input pointcloud is the global pointcloud
    extract_voxel_pointcloud.setInputCloud(cloud);
    // Extract points in the voxel (and not the rest of points in other voxels)
    extract_voxel_pointcloud.setNegative(false);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

    // K-d tree used for region growing
    pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    // Region growing class declaration
    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    // Cluster can contain 1 point
    reg.setMinClusterSize(1);
    // Cluster can contain every point of the current voxel
    reg.setMaxClusterSize(std::numeric_limits<int>::max());
    // Kd Tree used for region growing
    reg.setSearchMethod(tree);
    // Normals angle threshold set to 30 degrees
    reg.setSmoothnessThreshold(angle * M_PI / 180.0f);
    // Infinite curvature (if two points have normals angles difference less than 30 degrees => force them to be in the same cluster)
    reg.setCurvatureThreshold(std::numeric_limits<float>::max());

    // iterate over octree leaves

    for (it = octree.leaf_depth_begin(); it != it_end; ++it)
    {
        // Extract indices of input global pointcloud points which are in the i-th voxel

        // Declaration of points indices
        std::vector<int> indices;
        // Get indices of points in the voxel
        it.getLeafContainer().getPointIndices(indices);
        // Set indices of points in the voxel
        inliers->indices = indices;
        extract_voxel_pointcloud.setIndices(inliers);
        // Output pointcloud stored in voxel_cloud
        extract_voxel_pointcloud.filter(*voxel_cloud);
        pcl::copyPointCloud(*voxel_cloud, *voxel_cloud_xyz);
        pcl::copyPointCloud(*voxel_cloud, *voxel_cloud_normals);

        // Perform region growing clusterization in every voxel pointcloud
        // Every point of the voxel is considered as neighborhhod
        reg.setNumberOfNeighbours(voxel_cloud->size());
        // Set input cloud
        reg.setInputCloud(voxel_cloud_xyz);
        // Set input normals
        reg.setInputNormals(voxel_cloud_normals);
        // Extract each cluster obtained via region growing clusterization
        std::vector<pcl::PointIndices> clusters;
        reg.extract(clusters);

        // Test if more than one cluster was found. If this the case, iterate over clusters and commpute centroids
        if (clusters.size() != 1)
        {
            // Declare centroid of cluster
            pcl::PointNormal cluster_centroid;
            // Declare pointcloud associated to each cluster
            pcl::PointCloud<pcl::PointNormal>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointNormal>);
            // Extract cluster pointcloud class
            pcl::ExtractIndices<pcl::PointNormal> extract_cluster_pointcloud;
            pcl::PointIndices::Ptr inliers_cluster(new pcl::PointIndices());
            extract_cluster_pointcloud.setInputCloud(voxel_cloud);
            extract_cluster_pointcloud.setNegative(false);
            // Iterate over clusters of current voxel to compute spatial and normals centroid (use of PCL centroidPoint class)
            for (size_t i = 0; i < clusters.size(); i++)
            {
                inliers_cluster->indices = clusters[i].indices;
                extract_cluster_pointcloud.setIndices(inliers_cluster);
                extract_cluster_pointcloud.filter(*cluster_cloud);
                // Centroid class declaration
                pcl::CentroidPoint<pcl::PointNormal> centroid;
                // Iterate over points of the cluster to average the centroid position and normals
                for (size_t j = 0; j < cluster_cloud->size(); j++)
                {
                    centroid.add(cluster_cloud->points[j]);
                }
                // Compute centroid of the cluster
                centroid.get(cluster_centroid);
                // Model downsampling special case
                if (model == 1)
                {
                    // Pushback in dedicated vector (global variable) the number of points which have generated the centroid
                    modelWight.push_back((float)centroid.getSize() / cloud->size());
                    //  modelWight.push_back((float)centroid.getSize());
                }
                // Put it in the output_cloud
                output_cloud->push_back(cluster_centroid);
            }
        }
        // One cluster found in current voxel (which means every points in the voxel are in the same cluster [~=no sharp edges found, voxel_cloud=cluster_cloud])
        else
        {
            // Declare centroid of cluster
            pcl::PointNormal voxel_centroid;
            // Centroid class declaration
            pcl::CentroidPoint<pcl::PointNormal> centroid;
            // Iterate over points of the cluster to average the centroid position and normals
            for (size_t j = 0; j < voxel_cloud->size(); j++)
            {
                centroid.add(voxel_cloud->points[j]);
            }
            // Compute centroid of the cluster
            centroid.get(voxel_centroid);
            // Model downsampling special case
            if (model == 1)
            {
                // Pushback in dedicated vector (global variable) the number of points which have generated the centroid
                modelWight.push_back((float)centroid.getSize() / cloud->size());
                //modelWight.push_back((float)centroid.getSize());
            }
            // Put it in the output_cloud
            output_cloud->push_back(voxel_centroid);
        }
    }
    PCL_INFO("Cloud dimensions before / after subsampling: %u / %u\n", cloud->points.size(), output_cloud->points.size());
    return output_cloud;
}


PCLTOOL_NAMESPACE_END
