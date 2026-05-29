set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

include_directories(${CMAKE_SOURCE_DIR}/include)

add_subdirectory(pointwise)
add_subdirectory(reduce)
