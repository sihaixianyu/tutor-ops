cmake_minimum_required(VERSION 3.28)

project(cuda_add LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(cuda_add STATIC
    src/add.cu
)

target_include_directories(cuda_add
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(cuda_add PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)

add_executable(test_add
    test/test_add.cu
)

target_link_libraries(test_add
    PRIVATE
        cuda_add
)

set_target_properties(test_add PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)

add_test(NAME test_add COMMAND test_add)
set_tests_properties(test_add PROPERTIES SKIP_RETURN_CODE 77)
