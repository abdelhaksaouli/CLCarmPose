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

// C++ / C
#include <cstdio>
#include <cstdlib>

//Boost compute headers
#include <boost/compute/container/vector.hpp>

#include "defines.h"

#define HIST_NAMESPACE_BEGIN namespace clHistogram{
#define HIST_NAMESPACE_END }

HIST_NAMESPACE_BEGIN

template <typename Vector1, typename Vector2, typename Vector3>
void histogram(const Vector1& input, Vector2& histogram_values, Vector3& histogram_counts) 
{
    typedef typename Vector1::value_type ValueType; // input value type
    typedef typename Vector3::value_type IndexType; // histogram index type
    
    auto& clMangerInstance = CLContextManger::getInstance();

    size_t input_size = input.size();

    boostCL::vector<ValueType> data(input_size, clMangerInstance.context);
    std::vector<ValueType> host_data_temp(input_size); // i copy the input here and copy it back to device
    
    boostCL::copy(input.begin(), input.end(), host_data_temp.begin(), clMangerInstance.queue);
    boostCL::copy(host_data_temp.begin(), host_data_temp.end(), data.begin(), clMangerInstance.queue);
    host_data_temp.resize(0);

    IndexType num_bins = boostCL::inner_product(data.begin(), data.end() - 1,
                                                data.begin() + 1,
                                                IndexType(1),
                                                boostCL::plus<IndexType>(),
                                                boostCL::not_equal_to<ValueType>(), clMangerInstance.queue);

    histogram_values.resize(num_bins, clMangerInstance.queue);
    histogram_counts.resize(num_bins, clMangerInstance.queue);

    std::cout << "->number of bins : " << num_bins << std::endl;
    
    boostCL::reduce_by_key(data.begin(), data.end(),
                           boostCL::constant_iterator<IndexType>(1),
                           histogram_values.begin(),
                           histogram_counts.begin(), clMangerInstance.queue);

  /*  std::vector<Vector2::value_type> x(num_bins);
    std::vector<Vector3::value_type> y(num_bins);

    boostCL::copy(histogram_values.begin(), histogram_values.end(), x.begin(), clMangerInstance.queue);
    boostCL::copy(histogram_counts.begin(), histogram_counts.end(), y.begin(), clMangerInstance.queue);

    for (int i = 0; i < num_bins; i++)
    {
        std::cout << x[i] << std::endl;
    }

    std::cout << " " << std::endl;
    for (int i = 0; i < num_bins ; i++)
    {
        std::cout << y[i] << std::endl;
    }*/
}

HIST_NAMESPACE_END