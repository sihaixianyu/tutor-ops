add_executable(sum sum.cu)
target_link_libraries(sum PRIVATE fmt::fmt)

add_executable(softmax softmax.cu)
target_link_libraries(softmax PRIVATE fmt::fmt)
