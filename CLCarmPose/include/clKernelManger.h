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

//C/C++
#include <map>
#include <vector>
#include <string>

class clKernelManger
{

	public:
		static clKernelManger& getInstance()
		{
			static clKernelManger instance;

			return instance;
		}

		clKernelManger(const clKernelManger&) = delete;
		void operator=(clKernelManger const&) = delete;

	private:
		clKernelManger();

	public:
		std::map<std::string, boostCL::kernel> kernels;
	private:
		size_t block_size;
		size_t max_blocks;
	private:
		std::string loadCLProgram(const std::string& filePath);

	public:
		void set_block_size(const size_t& s);
		void add_kernel(const std::string& filePath, const std::string& name);
		void execute(const std::string& name, size_t data_size);

};

