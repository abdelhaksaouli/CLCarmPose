#include "clKernelManger.h"
#include "clContextManger.h"
#include "timer.h"

#include <fstream>
#include <algorithm>

clKernelManger::clKernelManger()
{
    block_size = 32;    // 32 thread per block
    max_blocks = 1024; //  max of 1024 block in grid
}

void clKernelManger::set_block_size(const size_t& s)
{
    block_size = s;
}

std::string clKernelManger::loadCLProgram(const std::string& filePath) 
{
    std::ifstream stream(filePath.c_str());
    if (!stream.is_open()) {
        std::cout << "Cannot open file: " << filePath << std::endl;
        exit(1);
    }
    return    std::string(
        std::istreambuf_iterator<char>(stream),
        (std::istreambuf_iterator<char>()));
}

void clKernelManger::add_kernel(const std::string& filePath, const std::string& name)
{
    auto& clMangerInstance = CLContextManger::getInstance();

    // get source file of cl kernel
    std::string source = loadCLProgram(filePath);

    //build the program
    boostCL::program clProgram = boostCL::program::create_with_source(source, clMangerInstance.context);

    std::cout << "->Build kernel from source : " << filePath << std::endl;

    try {
        clProgram.build();
    }
    catch (boostCL::opencl_error& e) {
        std::cout << clProgram.build_log() << std::endl;
    }
    // attach kernel to program
    try {

        boostCL::kernel  temp_kernel = clProgram.create_kernel(name);
        kernels.insert(std::make_pair(name, temp_kernel));
    }
    catch (boostCL::opencl_error& e) {
            std::cerr
                << "ERROR: "
                << e.what()
                << std::endl;
     }
}


void clKernelManger::execute(const std::string& name, size_t data_size)
{
    auto& clMangerInstance = CLContextManger::getInstance();
    try
    {
        std::cout << "->Kernel " << kernels[name].name() << " started" << std::endl;
        size_t blocks = std::min(((data_size + block_size) - 1) / block_size, max_blocks);
        size_t global = blocks * block_size;
        size_t local = block_size;

        boostCL::event evn = clMangerInstance.queue.enqueue_1d_range_kernel(kernels[name], 0, global, local);
        evn.wait();
        clMangerInstance.queue.finish();
        std::cout << "->Kernel " << kernels[name].name() << " took ";
        cltime::timing(evn);

    }
    catch (boostCL::opencl_error& e) {
        std::cerr
            << "ERROR: "
            << e.what()
            << std::endl;
    }
}