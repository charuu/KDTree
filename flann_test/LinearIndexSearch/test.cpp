
#include <flann/flann.hpp>
#include <flann/io/hdf5.h>

#include <stdio.h>

using namespace flann;

class FLANNTestFixture  {
protected:
    clock_t start_time_;
public:
    void start_timer(const std::string& message = "")
    {
        if (!message.empty()) {
            printf("%s", message.c_str());
            fflush(stdout);
        }
        start_time_ = clock();
    }

    double stop_timer()
    {
        return double(clock()-start_time_)/CLOCKS_PER_SEC;
    }

};

int main(int argc, char** argv)
{
	
    int nn = 3;
    FLANNTestFixture f;
    flann::Matrix<float> data;
    Matrix<float> query;
    const int n_points=10000;
    printf("creating random point cloud (%d points)...", n_points);
    data = flann::Matrix<float>(new float[n_points*3], n_points, 1);
    srand(1);
    for( int i=0; i<n_points; i++ )
    {
        data[i][0]=rand()/float(RAND_MAX);
        data[i][1]=rand()/float(RAND_MAX);
        data[i][2]=rand()/float(RAND_MAX);
       // std::cout<<data[i][0]<<" "<<data[i][1]<<" "<<data[i][2]<<std::endl;
    }
    Matrix<int> indices(new int[data.rows*nn], data.rows, nn);
    Matrix<float> dists(new float[data.rows*nn], data.rows, nn);
    std::cout << query.cols <<"\n";
    // construct an randomized kd-tree index using 4 kd-trees
    Index<L2<float> > index(data, flann::LinearIndexParams());
    index.buildIndex();                                                                                               

    // do a knn search, using 128 checks
    float r = 0.01;
    ///r=0.05;
	f.start_timer("Building kd-tree index...");
    index.radiusSearch( data, indices,dists, r*r, flann::SearchParams() );
 for( int i=0; i<dists.rows; i++ )
 		{
 			std::cout<<i<<" :" <<data[i][0]<< " " <<indices[i][0]<<" "<<dists[i][0]<<std::endl;
 		}

    std::cout<<"done (seconds)\n" <<f.stop_timer() <<"\n";

 //   flann::save_to_file(indices,"result.dat","result");

    delete[] data.ptr();
    delete[] query.ptr();
    delete[] indices.ptr();
    delete[] dists.ptr();
    
    return 0;
}
