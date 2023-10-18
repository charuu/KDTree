#include<stdio.h>
#define FLANN_USE_CUDA
#include<stdlib.h>
#include <time.h>
#include <string.h>
#include <flann/flann.h>
#include "../include/test_cuda.h"
#include <iostream>
using namespace std;
extern "C" flann_index_t call_func_build_tree(double* d){
        float *f = (float *) calloc(50000*3,sizeof(float));
    for(int i =0;i<50000*3;i++){
        f[i] = (float)d[i];
       // fprintf(stderr,"%lf ,%f\n",d[i],(float)d[i]);
    }
   return wrapper_build_tree(f);
}

extern "C" void call_func(double* d,int *a,flann_index_t indx){
         float *f = (float *) calloc(50000*3,sizeof(float));
    for(int i =0;i<50000*3;i++){
        f[i] = (float)d[i];
       // fprintf(stderr,"%lf ,%f\n",d[i],(float)d[i]);
    }
    wrapper_kernel_radius(f,a,indx); 
}