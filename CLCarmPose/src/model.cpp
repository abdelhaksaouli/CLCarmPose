#include "model.h"

#include "clContextManger.h"
#include "clKernelManger.h"
#include "histogram.h"
//boost compute headers
#include <boost/compute/container/vector.hpp>

Model::~Model()
{
	delete vote_counts_out;
}

Model::Model(pcl::PointCloud<pcl::PointNormal>* cloud, std::vector<float_> modelWights,
    float_ d_dist, float_ vote_count_threshold,
    uint_ use_l1_norm, uint_ use_averaged_clusters)
{
	std::cout << "Initialization Model creation from given point cloud with size of:" << cloud->size() << std::endl;
	std::vector<float4_>* points = new std::vector<float4_>(cloud->size());
	std::vector<float4_>* normals = new std::vector<float4_>(cloud->size());
	this->d_dist = d_dist;
	this->vote_count_threshold = vote_count_threshold;
	this->use_l1_norm = use_l1_norm;
	this->use_averaged_clusters = use_averaged_clusters;
	for (int i = 0; i < cloud->size(); i++)
	{
		(*points)[i][0] = (*cloud)[i].x;
		(*points)[i][1] = (*cloud)[i].y;
		(*points)[i][2] = (*cloud)[i].z;
		(*normals)[i][0] = (*cloud)[i].normal_x;
		(*normals)[i][1] = (*cloud)[i].normal_y;
		(*normals)[i][2] = (*cloud)[i].normal_z;

		(*points)[i][3] = 1.0f;
		(*normals)[i][3] = 0.0f;
	}
	if (points == nullptr || points->size() == 0 || normals == nullptr || normals->size() == 0)
	{
		std::cout << "->Something went wrong during  scene creation !!" << std::endl;
		return;
	}
	std::cout << "->Model Point cloud has been copyed to a host vectors (normal and points) with size of " << points->size()
		<< std::endl;
	initPPFs(points, normals, (int_)cloud->size(), d_dist);
	delete points;
	delete normals;
	
	auto& clMangerInstance = CLContextManger::getInstance();
	auto& clKernelMangerInstance = clKernelManger::getInstance();


	// set up wieghts
	modelPointVoteWeights = new boostCL::vector<float_>(modelWights.size(), clMangerInstance.context);
	boostCL::copy(modelWights.begin(), modelWights.end(), modelPointVoteWeights->begin(), clMangerInstance.queue);

	//hash model and save it in gpu hash structure
	boostCL::vector<uint_>* nonunique_hashkeys = new boostCL::vector<uint_>(modelPPFs->size(), clMangerInstance.context);
	// set up the kernel paramters
	try
	{
		clKernelMangerInstance.kernels["ppfHashKernel"].set_arg(0, this->modelPPFs->get_buffer());
		clKernelMangerInstance.kernels["ppfHashKernel"].set_arg(1, nonunique_hashkeys->get_buffer());
		clKernelMangerInstance.kernels["ppfHashKernel"].set_arg(2, this->n * this->n); // if there is problem check this
	}
	catch (boostCL::opencl_error& e) {
		std::cerr
			<< "ERROR: "
			<< e.what()
			<< std::endl;
	}
	// launch kernel
	clKernelMangerInstance.execute("ppfHashKernel", (size_t)this->n * this->n);

	//put the result into hash structure
	search_array = clParallelHashArray<uint_>(*nonunique_hashkeys);
}

void Model::ComputeUniqueVotes(Scene* scene)
{
	auto& clMangerInstance = CLContextManger::getInstance();
	auto& clKernelMangerInstance = clKernelManger::getInstance();

	boostCL::vector<size_t>* sceneIndices = search_array.GetIndices(*(scene->getHashKeys()));
	boostCL::vector<ulong_>* ppf_vote_counts = new boostCL::vector<ulong_>(scene->getHashKeys()->size(),
																		   clMangerInstance.context);
	try {
			clKernelMangerInstance.kernels["ppfVoteCountKernel"].set_arg(0, scene->getHashKeys()->get_buffer());
			clKernelMangerInstance.kernels["ppfVoteCountKernel"].set_arg(1, sceneIndices->get_buffer());
			clKernelMangerInstance.kernels["ppfVoteCountKernel"].set_arg(2, search_array.GetHashkeys()->get_buffer());
			clKernelMangerInstance.kernels["ppfVoteCountKernel"].set_arg(3, search_array.GetCounts()->get_buffer());
			clKernelMangerInstance.kernels["ppfVoteCountKernel"].set_arg(4, ppf_vote_counts->get_buffer());
			clKernelMangerInstance.kernels["ppfVoteCountKernel"].set_arg(5, (int_)scene->getHashKeys()->size());
		}
	catch (boostCL::opencl_error& e) {
		std::cerr
			<< "ERROR: "
			<< e.what()
			<< std::endl;
	}
	// launch kernel
	clKernelMangerInstance.execute("ppfVoteCountKernel", (size_t)scene->getHashKeys()->size());


	boostCL::vector<ulong_>* ppf_vote_indices = new boostCL::vector<ulong_>(scene->getHashKeys()->size(),clMangerInstance.context);
	boostCL::exclusive_scan(ppf_vote_counts->begin(), ppf_vote_counts->end(), ppf_vote_indices->begin(), clMangerInstance.queue);

	// copy data back to host, i need only the first value

	std::vector<ulong_>* host_ppf_vote_counts = new std::vector<ulong_>(1);
	std::vector<ulong_>* host_ppf_vote_indices = new std::vector<ulong_>(1);

	boostCL::copy_n(boost::prior(ppf_vote_counts->end()), 1, host_ppf_vote_counts->begin(), clMangerInstance.queue);
	boostCL::copy_n(boost::prior(ppf_vote_indices->end()), 1, host_ppf_vote_indices->begin(), clMangerInstance.queue);

	size_t num_votes = (*host_ppf_vote_counts)[0] + (*host_ppf_vote_indices)[0];
	std::cout <<"->number of non-unique votes is: "<< num_votes << std::endl;

	delete ppf_vote_counts;
	delete host_ppf_vote_counts;
	delete host_ppf_vote_indices;

	boostCL::vector<ulong_>* nonunique_nonempty_votes = new boostCL::vector<ulong_>(num_votes, clMangerInstance.context);
	try {
		clKernelMangerInstance.kernels["ppfVoteKernel"].set_arg(0, scene->getHashKeys()->get_buffer());
		clKernelMangerInstance.kernels["ppfVoteKernel"].set_arg(1, sceneIndices->get_buffer());
		clKernelMangerInstance.kernels["ppfVoteKernel"].set_arg(2, search_array.GetHashkeys()->get_buffer());
		clKernelMangerInstance.kernels["ppfVoteKernel"].set_arg(3, search_array.GetCounts()->get_buffer());
		clKernelMangerInstance.kernels["ppfVoteKernel"].set_arg(4, search_array.GetFirstHashkeyIndices()->get_buffer());
		clKernelMangerInstance.kernels["ppfVoteKernel"].set_arg(5, search_array.GetHashkeyToDataMap()->get_buffer());
		clKernelMangerInstance.kernels["ppfVoteKernel"].set_arg(6, modelPoints->get_buffer());
		clKernelMangerInstance.kernels["ppfVoteKernel"].set_arg(7, modelNormals->get_buffer());
		clKernelMangerInstance.kernels["ppfVoteKernel"].set_arg(8, (int_)this->n);
		clKernelMangerInstance.kernels["ppfVoteKernel"].set_arg(9, scene->getModelPoints()->get_buffer());
		clKernelMangerInstance.kernels["ppfVoteKernel"].set_arg(10, scene->getModelNormals()->get_buffer());
		clKernelMangerInstance.kernels["ppfVoteKernel"].set_arg(11, scene->numPoints());
		clKernelMangerInstance.kernels["ppfVoteKernel"].set_arg(12, ppf_vote_indices->get_buffer());
		clKernelMangerInstance.kernels["ppfVoteKernel"].set_arg(13, nonunique_nonempty_votes->get_buffer());
		clKernelMangerInstance.kernels["ppfVoteKernel"].set_arg(14, (int_)scene->getHashKeys()->size());
		clKernelMangerInstance.kernels["ppfVoteKernel"].set_arg(15, d_dist);
	}
	catch (boostCL::opencl_error& e) {
		std::cerr
			<< "ERROR: "
			<< e.what()
			<< std::endl;
	}
	// launch kernel
	clKernelMangerInstance.execute("ppfVoteKernel", (size_t)scene->getHashKeys()->size());
	delete ppf_vote_indices;
	boostCL::sort(nonunique_nonempty_votes->begin(), nonunique_nonempty_votes->end(), clMangerInstance.queue);
	std::cout << "->Sorting nonunique nonempty votes done " << std::endl;
	votes = new boostCL::vector<ulong_>(1, clMangerInstance.context);
	voteCounts = new boostCL::vector<uint_>(1, clMangerInstance.context);
	clHistogram::histogram(*nonunique_nonempty_votes, *votes, *voteCounts);
	std::cout<<"->number of unique votes: "<<votes->size()<<std::endl;

	boostCL::sort_by_key(voteCounts->begin(), voteCounts->end(), this->votes->begin(), 
						 boostCL::greater<float>(),
						 clMangerInstance.queue);

	std::vector<uint_> temp_votecounts(voteCounts->size());
	// copy data to cpu
	boostCL::copy(voteCounts->begin(), voteCounts->end(), temp_votecounts.begin(), clMangerInstance.queue);

	for (int i = 0; i < 10; i += 2)
		std::cout << i << ": " << temp_votecounts[i] << ", ";
	float min_votecount = vote_count_threshold * temp_votecounts[0];
	/* vote_count_threshold = (5 / modelSize ) * sum of sampling rates[Jerom 2020] */  /* temp_votecounts[0];*/
	size_t num_top_votes = boostCL::count_if(voteCounts->begin(), 
											 voteCounts->end(),
		                                     boostCL::bind(boostCL::greater<uint_>(), _1,(uint_) min_votecount),
											 clMangerInstance.queue);

	 std::cout<<std::endl<<"->num top votes:  "<< num_top_votes<<std::endl;

	this->votes->resize(num_top_votes, clMangerInstance.queue);
	this->voteCounts->resize(num_top_votes, clMangerInstance.queue);
}

boostCL::vector<float_>* Model::ComputeWeightedVoteCounts(boostCL::vector<ulong_>& votes,
													      boostCL::vector<uint_>& vote_counts,
														  boostCL::vector<float_>& modelpoint_vote_weights)
{
	auto& clMangerInstance = CLContextManger::getInstance();
	auto& clKernelMangerInstance = clKernelManger::getInstance();

	boostCL::vector<float_>* weighted_vote_counts = new boostCL::vector<float_>(votes.size(), clMangerInstance.context);
	try {

		clKernelMangerInstance.kernels["voteWeightKernel"].set_arg(0, votes.get_buffer());
		clKernelMangerInstance.kernels["voteWeightKernel"].set_arg(1, vote_counts.get_buffer());
		clKernelMangerInstance.kernels["voteWeightKernel"].set_arg(2, modelpoint_vote_weights.get_buffer());
		clKernelMangerInstance.kernels["voteWeightKernel"].set_arg(3, weighted_vote_counts->get_buffer());
		clKernelMangerInstance.kernels["voteWeightKernel"].set_arg(4, (int_)votes.size());
	}
	catch (boostCL::opencl_error& e) {
		std::cerr
			<< "ERROR: "
			<< e.what()
			<< std::endl;
	}
	// launch kernel
	clKernelMangerInstance.execute("voteWeightKernel", (size_t)votes.size());
	return weighted_vote_counts;
}


void Model::ComputeTransformations(Scene* scene)
{
	auto& clMangerInstance = CLContextManger::getInstance();
	auto& clKernelMangerInstance = clKernelManger::getInstance();

	transformations = new boostCL::vector<float_>(votes->size() * 16, clMangerInstance.context);

	try
	{
		clKernelMangerInstance.kernels["transCalcKernel"].set_arg(0, votes->get_buffer());
		clKernelMangerInstance.kernels["transCalcKernel"].set_arg(1, modelPoints->get_buffer());
		clKernelMangerInstance.kernels["transCalcKernel"].set_arg(2, modelNormals->get_buffer());
		clKernelMangerInstance.kernels["transCalcKernel"].set_arg(3, scene->getModelPoints()->get_buffer());
		clKernelMangerInstance.kernels["transCalcKernel"].set_arg(4, scene->getModelNormals()->get_buffer());
		clKernelMangerInstance.kernels["transCalcKernel"].set_arg(5, transformations->get_buffer());
		clKernelMangerInstance.kernels["transCalcKernel"].set_arg(6, (int_)this->votes->size());
	}
	catch (boostCL::opencl_error& e) {
		std::cerr
			<< "ERROR: "
			<< e.what()
			<< std::endl;
	}
	// launch kernel
	clKernelMangerInstance.execute("transCalcKernel", (size_t)votes->size());

}


boostCL::vector<float_>* Model::ClusterTransformations()
{
	auto& clMangerInstance = CLContextManger::getInstance();
	auto& clKernelMangerInstance = clKernelManger::getInstance();

	// set up some data as alwz
	transformation_trans = new boostCL::vector<float4_>(votes->size(), clMangerInstance.context);
	transformation_rots  = new boostCL::vector<float4_>(votes->size(), clMangerInstance.context);

	try
	{

		clKernelMangerInstance.kernels["mat2transquatKernel"].set_arg(0, transformations->get_buffer());
		clKernelMangerInstance.kernels["mat2transquatKernel"].set_arg(1, transformation_trans->get_buffer());
		clKernelMangerInstance.kernels["mat2transquatKernel"].set_arg(2, transformation_rots->get_buffer());
		clKernelMangerInstance.kernels["mat2transquatKernel"].set_arg(3, (int_)votes->size());
	}
	catch (boostCL::opencl_error& e) {
		std::cerr
			<< "ERROR: "
			<< e.what()
			<< std::endl;
	}
	// launch kernel
	clKernelMangerInstance.execute("mat2transquatKernel", (size_t)votes->size());
	/**********************************************************************************************************/
	boostCL::vector<uint_>* nonunique_trans_hash = new boostCL::vector<uint_>(votes->size(), clMangerInstance.context);
	boostCL::vector<uint_>* adjacent_trans_hash  = new boostCL::vector<uint_>(27 * votes->size(), clMangerInstance.context);

	try
	{
		clKernelMangerInstance.kernels["trans2idxKernel"].set_arg(0, transformations->get_buffer());
		clKernelMangerInstance.kernels["trans2idxKernel"].set_arg(1, nonunique_trans_hash->get_buffer());
		clKernelMangerInstance.kernels["trans2idxKernel"].set_arg(2, adjacent_trans_hash->get_buffer());
		clKernelMangerInstance.kernels["trans2idxKernel"].set_arg(3, (int_)votes->size());
		clKernelMangerInstance.kernels["trans2idxKernel"].set_arg(4, d_dist);
	}
	catch (boostCL::opencl_error& e) {
		std::cerr
			<< "ERROR: "
			<< e.what()
			<< std::endl;
	}
	// launch kernel
	clKernelMangerInstance.execute("trans2idxKernel", (size_t)votes->size());
	clParallelHashArray<uint_> trans_search_array = clParallelHashArray<uint_>(*nonunique_trans_hash);

	/*******************************************************************************************************************/
	boostCL::vector<size_t>* transIndices = trans_search_array.GetIndices(*adjacent_trans_hash);
	boostCL::vector<float_>* vote_counts_out = new boostCL::vector<float_>(weightedVoteCounts->size(), clMangerInstance.context);

	try
	{
		clKernelMangerInstance.kernels["rotClusteringKernel"].set_arg(0, transformation_trans->get_buffer());
		clKernelMangerInstance.kernels["rotClusteringKernel"].set_arg(1, transformation_rots->get_buffer());
		clKernelMangerInstance.kernels["rotClusteringKernel"].set_arg(2, weightedVoteCounts->get_buffer());
		clKernelMangerInstance.kernels["rotClusteringKernel"].set_arg(3, adjacent_trans_hash->get_buffer());
		clKernelMangerInstance.kernels["rotClusteringKernel"].set_arg(4, transIndices->get_buffer());
		clKernelMangerInstance.kernels["rotClusteringKernel"].set_arg(5, trans_search_array.GetHashkeys()->get_buffer());
		clKernelMangerInstance.kernels["rotClusteringKernel"].set_arg(6, trans_search_array.GetCounts()->get_buffer());
		clKernelMangerInstance.kernels["rotClusteringKernel"].set_arg(7,
																	  trans_search_array.GetFirstHashkeyIndices()->get_buffer());
		clKernelMangerInstance.kernels["rotClusteringKernel"].set_arg(8, trans_search_array.GetHashkeyToDataMap()->get_buffer());
		clKernelMangerInstance.kernels["rotClusteringKernel"].set_arg(9, vote_counts_out->get_buffer());
		clKernelMangerInstance.kernels["rotClusteringKernel"].set_arg(10, (int_)votes->size());
		clKernelMangerInstance.kernels["rotClusteringKernel"].set_arg(11, d_dist);
		clKernelMangerInstance.kernels["rotClusteringKernel"].set_arg(12, use_l1_norm);
		clKernelMangerInstance.kernels["rotClusteringKernel"].set_arg(13, use_averaged_clusters);
	}
	catch (boostCL::opencl_error& e) {
		std::cerr
			<< "ERROR: "
			<< e.what()
			<< std::endl;
	}
	// launch kernel
	clKernelMangerInstance.execute("rotClusteringKernel", (size_t)votes->size());
	delete transIndices;
	return vote_counts_out;
}

void Model::PPFEstimate(Scene* scene)
{
	auto& clMangerInstance = CLContextManger::getInstance();
	std::cout<< "*****************************************************************************************" <<std::endl;
	std::cout<< "                             Start looking up the scene in model                         " <<std::endl;
	ComputeUniqueVotes(scene);
	ComputeTransformations(scene);
	std::cout<<"->votes size: " << votes->size();
	weightedVoteCounts = ComputeWeightedVoteCounts(*votes, *voteCounts, *modelPointVoteWeights);
	vote_counts_out = ClusterTransformations();
	max_idx = boostCL::max_element(this->vote_counts_out->begin(), this->vote_counts_out->end(),
								   clMangerInstance.queue) - vote_counts_out->begin();
	std::cout << "->The transformation id is  :  " << this->max_idx << std::endl;
	std::cout<< "                             END of looking up the scene in model                         "<<std::endl;
	std::cout<< "*******************************************************************************************"<< std::endl;
}

Eigen::Matrix4f Model::getPose()const
{
	auto& clMangerInstance = CLContextManger::getInstance();
	Eigen::Matrix4f pose;

	std::vector<float_> *poses = new std::vector<float_>(transformations->size(), 0);
	boostCL::copy(transformations->begin(), transformations->end(), poses->begin(), clMangerInstance.queue);

	std::vector<float4_>* trans = new std::vector<float4_>(transformation_trans->size());
	boostCL::copy(transformation_trans->begin(), transformation_trans->end(), trans->begin(), clMangerInstance.queue);

	for (int r = 0; r < 4; r++) 
	{
		for (int c = 0; c < 4; c++) 
		{
			pose(r, c) = (*poses)[max_idx * 16 + r * 4 + c];
		}
	}
	pose(0, 3) = (*trans)[max_idx].x;
	pose(1, 3) = (*trans)[max_idx].y;
	pose(2, 3) = (*trans)[max_idx].z;
	delete poses;
	delete trans;

	return pose;
}