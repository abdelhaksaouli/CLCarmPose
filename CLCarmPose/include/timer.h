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


//boost compute
#include <boost/compute/event.hpp>

//C/C++
#include <iostream>
#include <chrono>

//api header
#include "defines.h"

#define CLTIME_NAMESPACE_BEGIN namespace cltime{
#define CLTIME_NAMESPACE_END }


CLTIME_NAMESPACE_BEGIN

/*****************************************************************
 * https://www.boost.org/doc/libs/1_67_0/boost/compute/event.hpp *
 *****************************************************************/

inline  void timing(boostCL::event& ev)
{
	std::cout << ev.duration<std::chrono::milliseconds>().count() << " ms" << std::endl;
}

CLTIME_NAMESPACE_END