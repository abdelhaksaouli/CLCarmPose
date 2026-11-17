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

// boost headers
#include <boost/compute/core.hpp>
#include <boost/compute/types/fundamental.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/algorithm.hpp>

// C/C++ headers
#include <iostream>

namespace boostCL = boost::compute;
/* cl types for host provided by the boost */
using boostCL::uint_;

using boostCL::int_;
using boostCL::ulong_;

using boostCL::float_;
using boostCL::float4_;

// to use this in count_if
using boostCL::placeholders::_1;
