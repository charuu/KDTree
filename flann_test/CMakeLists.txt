add_executable(test test.cpp)
target_link_libraries(test liblz4.a)
target_link_libraries(test flann)
set_target_properties(test PROPERTIES COMPILE_FLAGS -std=c++14)
include_directories(/opt/hdf5/include)
include_directories(/opt/hdf5/lib)

install (TARGETS test DESTINATION bin )