cmake_minimum_required(VERSION 3.20)

project(ops LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

find_package(fmt REQUIRED)

add_executable(vec_add src/add/vec_add.cu)
target_include_directories(vec_add PRIVATE include)
target_link_libraries(vec_add PRIVATE fmt::fmt)

add_executable(vec_add_float4 src/add/vec_add_float4.cu)
target_include_directories(vec_add_float4 PRIVATE include)
target_link_libraries(vec_add_float4 PRIVATE fmt::fmt)
