#include <time.h>
#define FLANN_USE_CUDA
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector_functions.h>
#include "../include/test_cuda.h"
#include <flann/util/matrix.h>

#include <stdlib.h>
#include <stdio.h>
#include <flann/flann.h>

extern "C" void wrapper_kernel_radius(float* point_arr,int *a,flann_index_t idx_ptr) {

    float r =0.25;
    int n_points=50000;
    int D=3;

    Matrix<int> m_indices(a, 50000, 1);
    flann::Matrix<float> query = flann:: Matrix<float>(point_arr, n_points, D);
    flann::Matrix<float> dists= flann::Matrix<float>(new float[n_points*1], n_points, 1);
    flann::KDTreeCuda3dIndex<L2_Simple<float>>* index =(flann::KDTreeCuda3dIndex<L2_Simple<float>>*)idx_ptr;
    
    thrust::host_vector<float4> data_host(n_points);
    for (int i = 0; i < query.rows; i++)
    {

        data_host[i] = make_float4(query[i][0], query[i][1], query[i][2], 0);
    }
    thrust::device_vector<float4> data_device = data_host;
    float4 b = data_device[2];
    std::cout<<"data:" << b.x << " "<< b.y << " " << b.z << " "<< b.w<< std::endl;
    
    
    
    
    
    index->radiusSearch(query, m_indices,dists, r*r, flann::SearchParams(-1) ); 
    return;
}
extern "C" flann_index_t wrapper_build_tree(float* point_arr) {
    int n_points=50000;
    int D=3;
    flann::Matrix<float> query = flann:: Matrix<float>(point_arr, n_points, D);
    thrust::host_vector<float4> query_host(query.rows);
    for( int i=0; i<query.rows; i++ )
	{
		query_host[i]=make_float4(query[i][0],query[i][1],query[i][2],0);
	}
    thrust::device_vector<float4> query_device = query_host;

    flann::Matrix<float> query_device_matrix( (float*)thrust::raw_pointer_cast(&query_device[0]),query.rows,3,4*4);
    flann::KDTreeCuda3dIndexParams index_params;
    index_params["input_is_gpu_float4"] = true;
   
    flann::KDTreeCuda3dIndex<L2<float> >* idx = new KDTreeCuda3dIndex<L2<float> >(query_device_matrix, index_params);
   
    idx->buildIndex();

    return (flann_index_t) idx ;

}
 