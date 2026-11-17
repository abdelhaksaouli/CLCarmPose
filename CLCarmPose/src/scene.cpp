#include "scene.h"
#include "clContextManger.h"
#include "clKernelManger.h"
//boost compute headers
#include <boost/compute/container/vector.hpp>

Scene::~Scene()
{
	delete modelPoints;
	delete modelNormals;
	delete modelPPFs;
}

Scene::Scene()
{
	//Empty
	n = 0;
	modelPoints  = nullptr;
	modelNormals = nullptr;
	modelPPFs    = nullptr;
	hashKeys     = nullptr;
	d_dist = 0.0f;
}

Scene::Scene(pcl::PointCloud<pcl::PointNormal>* cloud_ptr, float_ d_dist, float_ sceneRaduis, uint_ ref_point_downsample_factor)
{
	std::cout << "Initialization scene creation from given point cloud with size of:" << cloud_ptr->size() << std::endl;
	std::vector<float4_>* points = new std::vector<float4_>(cloud_ptr->size());
	std::vector<float4_>* normals = new std::vector<float4_>(cloud_ptr->size());
	this->d_dist = d_dist;
	for (int i = 0; i < cloud_ptr->size(); i++)
	{
		(*points)[i][0] = (*cloud_ptr)[i].x;
		(*points)[i][1] = (*cloud_ptr)[i].y;
		(*points)[i][2] = (*cloud_ptr)[i].z;
		(*normals)[i][0] = (*cloud_ptr)[i].normal_x;
		(*normals)[i][1] = (*cloud_ptr)[i].normal_y;
		(*normals)[i][2] = (*cloud_ptr)[i].normal_z;

		(*points)[i][3] = 1.0f;
		(*normals)[i][3] = 0.0f;
	}
	if (points == nullptr || points->size() == 0 || normals == nullptr || normals->size() == 0)
	{
		std::cout << "->Something went wrong during  scene creation !!" << std::endl;
     	return;
	}
	std::cout << "->Scene Point cloud has been copyed to a host vectors (normal and points) with size of " << points->size() 
		      << std::endl;
	initPPFs(points, normals, (int_)cloud_ptr->size(), d_dist, 1, ref_point_downsample_factor, sceneRaduis);
	delete points;
	delete normals;
	// hash the the scene
	auto& clMangerInstance = CLContextManger::getInstance();
	auto& clKernelMangerInstance = clKernelManger::getInstance();
	hashKeys = new boostCL::vector<uint_>(modelPPFs->size(), clMangerInstance.context);
	// set up the kernel paramters
	try
	{
		clKernelMangerInstance.kernels["ppfHashKernel"].set_arg(0, this->modelPPFs->get_buffer());
		clKernelMangerInstance.kernels["ppfHashKernel"].set_arg(1, this->hashKeys->get_buffer());
		clKernelMangerInstance.kernels["ppfHashKernel"].set_arg(2, this->n * this->n);
	}
	catch (boostCL::opencl_error& e) {
		std::cerr
			<< "ERROR: "
			<< e.what()
			<< std::endl;
	}
	// launch kernel
	clKernelMangerInstance.execute("ppfHashKernel", (size_t)this->n * this->n);
}


void Scene::initPPFs(std::vector<float4_>* points, std::vector<float4_>* normals, int_ n, float_ d_dist, int_ isScene, uint_						   ref_point_downsample_factor, float_ sceneRaduis)
{
	// instance it cl context
	auto& clMangerInstance		 = CLContextManger::getInstance();
	auto& clKernelMangerInstance = clKernelManger::getInstance();

	this->n = n;
	modelPoints = new boostCL::vector<float4_>(this->n, clMangerInstance.context);
	modelNormals = new boostCL::vector<float4_>(this->n, clMangerInstance.context);
	// transfer data from the host to the device
	boostCL::copy(points->begin(), points->end(),   modelPoints->begin(), clMangerInstance.queue);
	boostCL::copy(normals->begin(), normals->end(), modelNormals->begin(), clMangerInstance.queue);
	std::cout << "->Point cloud has been copyed to a device vectors (normal and points) with size of " << modelPoints->size() 
		      << std::endl;

	modelPPFs = new boostCL::vector<float4_>((size_t)(this->n*this->n), clMangerInstance.context);
	// set up the kernel paramters
	try
	{
		clKernelMangerInstance.kernels["ppfKernel"].set_arg(0, modelPoints->get_buffer());
		clKernelMangerInstance.kernels["ppfKernel"].set_arg(1, modelNormals->get_buffer());
		clKernelMangerInstance.kernels["ppfKernel"].set_arg(2, modelPPFs->get_buffer());
		clKernelMangerInstance.kernels["ppfKernel"].set_arg(3, (int)this->n);
		clKernelMangerInstance.kernels["ppfKernel"].set_arg(4, ref_point_downsample_factor);
		clKernelMangerInstance.kernels["ppfKernel"].set_arg(5, d_dist);
		clKernelMangerInstance.kernels["ppfKernel"].set_arg(6, isScene);
		clKernelMangerInstance.kernels["ppfKernel"].set_arg(7, sceneRaduis);
	}
	catch (boostCL::opencl_error& e) {
		std::cerr
			<< "ERROR: "
			<< e.what()
			<< std::endl;
	}
	// launch kernel
	clKernelMangerInstance.execute("ppfKernel", (size_t)n);
}

