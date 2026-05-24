add_executable(add add.cu)
target_link_libraries(add PRIVATE fmt::fmt)

add_executable(add_float4 add_float4.cu)
target_link_libraries(add_float4 PRIVATE fmt::fmt)
