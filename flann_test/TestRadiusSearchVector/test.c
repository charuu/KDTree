
#include<stdio.h>
#include<stdlib.h>
#include <time.h>
#include <string.h>
#define FLANN_USE_CUDA
#include <flann/flann.h>
#ifdef __cplusplus
extern "C" flann_index_t call_func_build_tree(double*);
extern "C" void  call_func(double*,int *,flann_index_t );
#endif
void read2(FILE* fp ,double* point_arr){
    
    char line[120];
    char* end;
    char* token;
    double val;
    int i =0;
    int D=3;
    while(fgets(line,120,fp)){
        int j=0;
        token  = strtok(line, "\t");
        while(token!=NULL){
            val = strtod(token,&end);
            *(point_arr)=val;
            token = strtok(NULL, "\t");
            point_arr++;
        };
       // point_arr++;
    } 
}

int main(){
   // float* d = calloc(7,sizeof(float));
   // d[0]=99;
    int n_points=50000;
    int D=3;
    double *point_arr= (double*) malloc(n_points*3*sizeof(double));
    FILE *fp_y = fopen("../datasets/YY_after.txt", "r");

    double *query_arr= (double*) malloc(n_points*3*sizeof(double));
    FILE *fp_x = fopen("../datasets/XX_after.txt", "r");

    int *a= (int*) malloc(n_points*sizeof(long int));
    read2(fp_x,point_arr);
    read2(fp_y,query_arr);
    fclose(fp_x);
    fclose(fp_y);

   //build kd tree
	flann_index_t idx = call_func_build_tree(point_arr);
    clock_t start_time , stop_time;
    start_time = clock();
    
    // find neighbiurs in indexed tree
    //for (int i=0;i<50;i++){
        call_func(query_arr,a,idx);
    //}

    for(int i=0;i<n_points;i++){
        printf("%d , %d\n", i,a[i]);
    }
    stop_time = clock();
    printf("%lf\n", ((double)stop_time - start_time)/CLOCKS_PER_SEC);
   
    return 0;
}