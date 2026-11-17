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
#pragma once

// C/C++ related headers 
#include <iostream>
#include <string>

// PCL related headers
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#define PCLOADER_NAMESPACE_BEGIN namespace loader {
#define PCLOADER_NAMESPACE_END }


PCLOADER_NAMESPACE_BEGIN

// simple function that load point cloud from pcd file
inline pcl::PointCloud<pcl::PointXYZ>::Ptr loadPointCloud(std::string filename)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcd(new pcl::PointCloud<pcl::PointXYZ>);

	if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *pcd) == -1)
	{
		return nullptr;
	}

	std::cout << "File loaded with (" << pcd->width << "*" << pcd->height << ") data points from " << filename << std::endl;

	return pcd;
}

inline pcl::PointCloud<pcl::PointNormal>::Ptr loadPointCloudWithNormal(std::string filename)
{
	pcl::PointCloud<pcl::PointNormal>::Ptr pcd(new pcl::PointCloud<pcl::PointNormal>);

	if (pcl::io::loadPCDFile<pcl::PointNormal>(filename, *pcd) == -1)
	{
		return nullptr;
	}

	std::cout << "File loaded with (" << pcd->width << "*" << pcd->height << ") data points from " << filename << std::endl;

	return pcd;
}
inline void savePCD(pcl::PointCloud<pcl::PointXYZ>::Ptr pcd ,std::string filename)
{
	pcl::io::savePCDFileASCII(filename, *pcd);

	std::cerr << "Saved " << pcd->size() << " data points" << std::endl;
}

// Print function
inline void printPoint(pcl::PointCloud<pcl::PointXYZ>::Ptr pcd)
{
	for (const auto& point : *pcd)
		std::cout << "    " << point.x << " " << point.y << " " << point.z << std::endl;
}

PCLOADER_NAMESPACE_END