#include "gpuppfstimator.h"
#include <fstream>

GPUPPFEstimator::GPUPPFEstimator()
{
	readParameters();
	model_d_dists = 0.0f;                   /* model discritization distance used in downsampling and in ppf discritization
											   its equal to t_au_d * model diam */
	scene_d_dists = 0.0;						/* scene discritization distance used only in downsampling  */
}

GPUPPFEstimator::~GPUPPFEstimator()
{

}

void GPUPPFEstimator::readParameters()
{
	std::ifstream script("data/estimate.txt", std::ifstream::in);
	if (!script)
	{
		std::cerr << " unable to open file!!!" << std::endl;
		exit(EXIT_FAILURE);
	}

	script >> model_files;
	script >> scene_files;
	//script >> scene_leaf_size;
	script >> ref_point_df;
	script >> tau_d;
	script >> downsamplingAngle;
	script >> picked_point.x >> picked_point.y >> picked_point.z;
	script.close();
}

void GPUPPFEstimator::init_device()
{
	enable_cuda_build_cache(false); // after debuging set it to true
	auto& clMangerInstance = CLContextManger::getInstance();
	clMangerInstance.printDeviceInfo();
	auto& clKernelMangerInstance = clKernelManger::getInstance();
	std::cout << "->Set up kernels......"<< std::endl;
	clKernelMangerInstance.add_kernel("kernels/ppf_hash_kernel.cl", "ppfHashKernel");
	clKernelMangerInstance.add_kernel("kernels/ppf_kernel.cl", "ppfKernel");
	clKernelMangerInstance.add_kernel("kernels/ppf_vote_kernel.cl", "ppfVoteKernel");
	clKernelMangerInstance.add_kernel("kernels/ppf_vote_count_kernel.cl", "ppfVoteCountKernel");
	clKernelMangerInstance.add_kernel("kernels/vote_weight_kernel.cl", "voteWeightKernel");
	clKernelMangerInstance.add_kernel("kernels/vec_lower_bound.cl", "lowerBound");
	clKernelMangerInstance.add_kernel("kernels/trans2idx_kernel.cl", "trans2idxKernel");
	clKernelMangerInstance.add_kernel("kernels/trans_calc_kernel.cl", "transCalcKernel");
	clKernelMangerInstance.add_kernel("kernels/rot_clustering_kernel.cl", "rotClusteringKernel");
	clKernelMangerInstance.add_kernel("kernels/mat2transquat_kernel.cl", "mat2transquatKernel");
	std::cout << "->Kernel programe loaded compiled.." << std::endl;

}

void GPUPPFEstimator::computeVoteCountThreshold()
{
	vote_count_threshold = 0.0;

	for (int_ i = 0; i < modelWight.size(); ++i)
		vote_count_threshold += modelWight[i];

	vote_count_threshold /= modelWight.size();
}

void GPUPPFEstimator::trainModel()
{
	std::cout << "##############################################################################" << std::endl;
	std::cout << "->Traning the model" << std::endl;
	pcl::PointCloud<pcl::PointNormal>::Ptr pcd_normal = loader::loadPointCloudWithNormal(model_files);
	model_d_dists     = tools::cloudDiam(pcd_normal) * tau_d;
	sceneSearchRaduis = tools::cloudDiam(pcd_normal) / 2.0;
	pcl::PointCloud<pcl::PointNormal>::Ptr DownsampledModel = tools::advance_voxel_grid_down_sample(pcd_normal, 
																									model_d_dists,																				   downsamplingAngle, 1,																		   modelWight);
	vote_count_threshold = 0.2f; // or compute it via computeVoteCountThreshold()
	cad = new Model(DownsampledModel.get(), modelWight, model_d_dists, vote_count_threshold, 0, 1);
	pcd_normal->resize(0);
	DownsampledModel->resize(0);
}

void GPUPPFEstimator::setUpScene()
{
	std::cout << "->Setup the scene" << std::endl;
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcd = tools::cpu_rand_downsample(loader::loadPointCloud(scene_files), 0.001);
	pcl::PointCloud<pcl::PointNormal>::Ptr pcd_normal = tools::compute_normal_unorganized(pcd);
	pcl::PointCloud<pcl::PointNormal>::Ptr selection = tools::object_picking(pcd_normal, picked_point);
	// its okey to use modelwieght again it will not change since this is scene not model
	pcl::PointCloud<pcl::PointNormal>::Ptr DownsampledScene = tools::advance_voxel_grid_down_sample(selection,
																									model_d_dists,
																									downsamplingAngle, 0,																		   modelWight);
	scn = new Scene(DownsampledScene.get(), model_d_dists, sceneSearchRaduis, ref_point_df);
	pcd->resize(0);
	pcd_normal->resize(0);
	DownsampledScene->resize(0);
	selection->resize(0);
}


void GPUPPFEstimator::PPFGPURegistration()
{
	
	std::cout << "->look the model" << std::endl;
	cad->PPFEstimate(scn);
	results = cad->getPose();
	delete scn;
	delete cad;
	std::ofstream pose("data/pose.txt", std::ofstream::out);
	if (!pose)
	{
		std::cerr << " unable to create file" << std::endl;
		exit(EXIT_FAILURE);
	}

	pose << results(0, 0) << " " << results(0, 1) << " " << results(0, 2) << " " << results(0, 3) << " "
		<< results(1, 0) << " " << results(1, 1) << " " << results(1, 2) << " " << results(1, 3) << " "
		<< results(2, 0) << " " << results(2, 1) << " " << results(2, 2) << " " << results(2, 3) << " "
		<< results(3, 0) << " " << results(3, 1) << " " << results(3, 2) << " " << results(3, 3) << std::endl;

	std::cout << results(0, 0) << " " << results(0, 1) << " " << results(0, 2) << " " << results(0, 3) << std::endl;
	std::cout << results(1, 0) << " " << results(1, 1) << " " << results(1, 2) << " " << results(1, 3) << std::endl;
	std::cout << results(2, 0) << " " << results(2, 1) << " " << results(2, 2) << " " << results(2, 3) << std::endl;
	std::cout << results(3, 0) << " " << results(3, 1) << " " << results(3, 2) << " " << results(3, 3) << std::endl;
	pose.close();
	std::cout << "##############################################################################" << std::endl;
}