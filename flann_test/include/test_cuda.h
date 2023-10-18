
#include <flann/flann.h>
extern "C" void wrapper_kernel_radius(float* query_arr,int *indices,flann_index_t idx) ;
extern "C" flann_index_t wrapper_build_tree(float* point_arr);