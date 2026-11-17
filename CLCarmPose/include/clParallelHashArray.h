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

#include "defines.h"
#include "clContextManger.h"
#include "clKernelManger.h"

#include "histogram.h"

//boost compute headers
#include <boost/compute/container/vector.hpp>


template<typename T>
class clParallelHashArray
{

    public:
        clParallelHashArray();
        clParallelHashArray(boostCL::vector<T>& data);

        boostCL::vector<size_t>* GetIndices(boostCL::vector<uint_>& data_hashkeys);

        boostCL::vector<uint_>* GetHashkeys();
        boostCL::vector<size_t>* GetCounts();
        boostCL::vector<size_t>* GetFirstHashkeyIndices();
        boostCL::vector<size_t>* GetHashkeyToDataMap();

        private:
            // Size of data.
            size_t n;

            // Indices into data.
            // nonunique_hashkeys[i] == hash(data[hashkeyToDataMap[i]])
            boostCL::vector<size_t> *hashkeyToDataMap;
            // *unique* hashkeys.
            boostCL::vector<uint_> *hashkeys;
            // number of occurances of each hashkey
            boostCL::vector<size_t> *counts;
            // Indices in hashkeyToDataMap where blocks of identical hashkeys begin.
            boostCL::vector<size_t> *firstHashkeyIndex;


};

template<typename T>
clParallelHashArray<T>::clParallelHashArray()
{
	//EMPTY
}


template<typename T>
clParallelHashArray<T>::clParallelHashArray(boostCL::vector<T>& data)
{
	auto& clMangerInstance = CLContextManger::getInstance();
	n = data.size();
	hashkeyToDataMap = new boostCL::vector<size_t>(n, clMangerInstance.context);
	// this the same as sequence in thrust (fill array with simple sequence start from 0 to size - 1)
	boostCL::iota(hashkeyToDataMap->begin(), hashkeyToDataMap->end(), 0, clMangerInstance.queue);
	// Sort nonunique_hashkeys and hashkeyToDataMap.
	boostCL::sort_by_key(data.begin(), data.end(), hashkeyToDataMap->begin(), clMangerInstance.queue);
	hashkeys = new boostCL::vector<uint_>(1, clMangerInstance.context);
	counts = new boostCL::vector<size_t>(1, clMangerInstance.context);
	clHistogram::histogram(data, *hashkeys, *counts);
	firstHashkeyIndex = new boostCL::vector<size_t>(hashkeys->size(), clMangerInstance.context);
	boostCL::exclusive_scan(counts->begin(), counts->end(), firstHashkeyIndex->begin(), clMangerInstance.queue);
	std::cout << "->Parallel Hash Array Created" << std::endl;
}


template <typename T>
boostCL::vector<uint_>* clParallelHashArray<T>::GetHashkeys()
{
	return this->hashkeys;
}

template <typename T>
boostCL::vector<size_t>* clParallelHashArray<T>::GetCounts()
{
	return this->counts;
}

template <typename T>
boostCL::vector<size_t>* clParallelHashArray<T>::GetFirstHashkeyIndices()
{
	return this->firstHashkeyIndex;
}

template <typename T>
boostCL::vector<size_t>* clParallelHashArray<T>::GetHashkeyToDataMap()
{
	return this->hashkeyToDataMap;
}


template <typename T>
boostCL::vector<size_t>* clParallelHashArray<T>::GetIndices(boostCL::vector<uint_>& data_hashkeys)
{
	auto& clMangerInstance = CLContextManger::getInstance();
	auto& clKernelMangerInstance = clKernelManger::getInstance();

	// find possible starting indices of blocks matching Model hashKeys
	// set up the out put indice vector on gpu
	boostCL::vector<size_t>* dataIndices = new boostCL::vector<size_t>(data_hashkeys.size(), clMangerInstance.context);
	//start the lower_bound binary search kernel
	// set up the kernel paramters
	try
	{
		clKernelMangerInstance.kernels["lowerBound"].set_arg(0, hashkeys->get_buffer());
		clKernelMangerInstance.kernels["lowerBound"].set_arg(1, data_hashkeys.get_buffer());
		clKernelMangerInstance.kernels["lowerBound"].set_arg(2, dataIndices->get_buffer());
		clKernelMangerInstance.kernels["lowerBound"].set_arg(3, (int)hashkeys->size());
		clKernelMangerInstance.kernels["lowerBound"].set_arg(4, (int)data_hashkeys.size());
	}
	catch (boostCL::opencl_error& e) {
		std::cerr
			<< "ERROR: "
			<< e.what()
			<< std::endl;
	}

	// launch kernel
	clKernelMangerInstance.execute("lowerBound", (size_t)data_hashkeys.size());
	return dataIndices;
}