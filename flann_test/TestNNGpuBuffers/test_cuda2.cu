
#include <time.h>
#define FLANN_USE_CUDA


#include <flann/util/matrix.h>
#include <vector>
#include <set>

#include<stdlib.h>
#include<stdio.h>
#include <flann/flann.h>
#include <flann/io/hdf5.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector_functions.h>

 void start_timer(const std::string& message = "")
    {
        clock_t start_time_;
        if (!message.empty()) {
            printf("%s", message.c_str());
            fflush(stdout);
        }
        start_time_ = clock();
    }

    double stop_timer()
    {
        clock_t start_time_;
        return double(clock()-start_time_)/CLOCKS_PER_SEC;
    }
int find_correspondences(){
    printf("Reading test data...");
    fflush(stdout);
    int nn=3;
    flann::Matrix<float> data;
    flann::Matrix<float> query;
    flann::Matrix<float> dists;
    flann::Matrix<int> indices;
    Matrix<int> match(new int[query.rows*nn], data.rows, nn);
    
    const int n_points=50000;
    printf("creating random point cloud (%d points)...", n_points);
    data = flann::Matrix<float>(new float[n_points*3], n_points,3);
    srand(1);
    FILE *fp = fopen("datasets/XX_after.txt", "r");
    int RANGE_MAX = 100;
    const int max_nn = 1;
		
    flann::Matrix<float> gt_dists;
    gt_dists = flann::Matrix<float>(new float[query.rows*max_nn], query.rows, max_nn);
   
    if (fp == NULL)
    {
        puts("Couldn't open file");
        exit(0);
    }
    int n =50000;
    int dim=3;
    char line[120];
    char* end;
    char* token;
    float val;
    float point_arr[50000][3];
    int i =0;
    
    while(fgets(line,120,fp)){
        int j=0;
        token  = strtok(line, "\t");
        while(token!=NULL){
            val = strtod(token,&end);
            point_arr[i][j]=val;
            //printf("%lf \t",val);
            token = strtok(NULL, "\t");
            j++;
        }
        i++;
      //   printf("\n");
    }
    
    for( int i=0; i<n_points; i++ )
    {
        data[i][0]=point_arr[i][0];//rand()/float(RAND_MAX);
        data[i][1]=point_arr[i][1];//rand()/float(RAND_MAX);
        data[i][2]=point_arr[i][2];//rand()/float(RAND_MAX);
        //   std::cout<<data[i][0]<<" "<<data[i][1]<<" "<<data[i][2]<<std::endl;
    }
    query = flann::Matrix<float>(new float[n_points*3], n_points,3);
    for( int i=0; i<n_points; i++ )
    {
        query[i][0]=point_arr[i][0];
        query[i][1]=point_arr[i][1];
        query[i][2]=point_arr[i][2];
        // std::cout<<data[i][0]<<" "<<data[i][1]<<" "<<data[i][2]<<std::endl;
    }
    thrust::host_vector<float4> data_host(data.rows);
	for( int i=0; i<data.rows; i++ )
	{
		data_host[i]=make_float4(data[i][0],data[i][1],data[i][2],0);
	}
	thrust::device_vector<float4> data_device = data_host;
	thrust::host_vector<float4> query_host(data.rows);
	for( int i=0; i<data.rows; i++ )
	{
		query_host[i]=make_float4(query[i][0],query[i][1],query[i][2],0);
	}
	thrust::device_vector<float4> query_device = query_host;
	
	flann::Matrix<float> data_device_matrix( (float*)thrust::raw_pointer_cast(&data_device[0]),data.rows,3,4*4);
	flann::Matrix<float> query_device_matrix( (float*)thrust::raw_pointer_cast(&query_device[0]),data.rows,3,4*4);
	
	flann::KDTreeCuda3dIndexParams index_params;
	index_params["input_is_gpu_float4"]=true;
	flann::KDTreeCuda3dIndex<L2_Simple<float> > index(data_device_matrix, index_params);
    start_timer("Building kd-tree index...");
    index.buildIndex();
    printf("done (%g seconds)\n", stop_timer());

	
	thrust::device_vector<int> indices_device(query.rows*4);
	thrust::device_vector<float> dists_device(query.rows*4);
	flann::Matrix<int> indices_device_matrix( (int*)thrust::raw_pointer_cast(&indices_device[0]),query.rows,4);
	flann::Matrix<float> dists_device_matrix( (float*)thrust::raw_pointer_cast(&dists_device[0]),query.rows,4);
	
    start_timer("Searching KNN...");
	indices.cols=4;
	dists.cols=4;
	flann::SearchParams sp;
	sp.matrices_in_gpu_ram=true;
    index.knnSearch(query_device_matrix, indices_device_matrix, dists_device_matrix, 4, sp );
    printf("done (%g seconds)\n", stop_timer());
	
	flann::Matrix<int> indices_host( new int[ query.rows*4],query.rows,4 );
	flann::Matrix<float> dists_host( new float[ query.rows*4],query.rows,4 );
	
	thrust::copy( dists_device.begin(), dists_device.end(), dists_host.ptr() );
	thrust::copy( indices_device.begin(), indices_device.end(), indices_host.ptr() );

    // float precision = computePrecisionDiscrete(gt_dists,dists_host, 1e-08);

    //printf("Precision: %g\n", precision);
	fclose(fp);
	delete [] indices_host.ptr();
	delete [] dists_host.ptr();
}

extern "C" void wrapper_kernel() {
     find_correspondences();
     return;
}