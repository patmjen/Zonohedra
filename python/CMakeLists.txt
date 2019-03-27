cmake_minimum_required(VERSION 3.11)

set(CMAKE_GENERATOR_PLATFORM x64)

project ("pygorpho")

find_package(CUDA)
message("Using CUDA version: " ${CUDA_VERSION_STRING})

set(CUDA_NVCC_FLAGS "--expt-relaxed-constexpr;-lineinfo")
cuda_add_library(pygorpho pygorpho_api.cu SHARED)

include_directories(..)
include_directories(../cudablockproc/lib)