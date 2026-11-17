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


/* Create a SINGLETON class that have the device context and queue of opencl
*  and use it for every thing since am counting on using one context for the
*  hole algorithm
*/

class CLContextManger
{
    public:
        static CLContextManger& getInstance()
        {
            static CLContextManger instance;

            return instance;
        }

        CLContextManger(const CLContextManger&) = delete;
        void operator=(CLContextManger const&) = delete;

    private:
        CLContextManger()
        {
            device = boostCL::system::default_device();
            context = boostCL::context(device);
            queue = boostCL::command_queue(context, device, CL_QUEUE_PROFILING_ENABLE);
         
        }

    public:
        boostCL::device  device;
        boostCL::context context;
        boostCL::command_queue queue;

    public:
        inline void printDeviceInfo()
        {
            std::cout << "******************************************************************** " << std::endl;
            std::cout << "Device name" << device.name() <<std::endl;
            std::cout << "Device vendor" << device.vendor() << std::endl;
            std::cout << "Device driver version " << device.driver_version() << std::endl;
            std::cout << "******************************************************************** " << std::endl;
        }
};

