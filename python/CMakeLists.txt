cmake_minimum_required(VERSION 2.8.0)

if (CMAKE_GENERATOR MATCHES "Visual Studio")
	# For Visual Studio we need to force 64 bit
	set(CMAKE_GENERATOR_PLATFORM x64)
endif()

project ("pygorpho")

find_package(CUDA)
message("Using CUDA version: " ${CUDA_VERSION_STRING})

include_directories(..)
include_directories(../cudablockproc/lib)

set(CUDA_NVCC_FLAGS "--expt-relaxed-constexpr;-lineinfo;--use_fast_math")
if(NOT MSVC)
	# It seems like Microsofts compiler selects a good C++ version per default,
	# but for others we specify it manually
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler='-std=c++14'")
endif()
cuda_add_library(pygorpho pygorpho_api.cu SHARED)