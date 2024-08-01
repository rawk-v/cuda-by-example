cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(CudaExample LANGUAGES CXX CUDA)

# Specify the CUDA toolkit version (optional)
find_package(CUDA REQUIRED)
# find cublas
find_library(CUBLAS_LIBRARY cublas HINTS /usr/local/cuda/lib64)

# Set the CUDA architecture (optional, specify based on your GPU)
# For example, sm_60 is for Pascal GPUs, sm_75 for Turing GPUs, sm_80 for Ampere GPUs.
# You can find the appropriate compute capability for your GPU on NVIDIA's website.
set(CMAKE_CUDA_ARCHITECTURES 60 75 80)

# Add the source file
set(SOURCE_FILES main.cu)

# Create an executable
add_executable(CudaExample ${SOURCE_FILES})

# add another executable
add_executable(cublas-gemm cublas-gemm.cu)

# Set the target properties
set_target_properties(CudaExample PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 11
        CXX_STANDARD 11
)

# Include CUDA headers
include_directories(${CUDA_INCLUDE_DIRS})

# Link CUDA libraries (optional, usually handled automatically)
target_link_libraries(CudaExample ${CUDA_LIBRARIES})
#link cublas to cublas-gemm
target_link_libraries(cublas-gemm ${CUBLAS_LIBRARY} ${CUDA_LIBRARIES})