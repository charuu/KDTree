
/usr/local/cuda-12.1/bin/nvcc -c -Xcompiler -fPIC  -Xcudafe --diag_suppress=partial_override --compiler-bindir=/usr/bin  test_cuda.cu -l:liblz4.a -std=c++17 test_cuda.o -arch=compute_89 -L /usr/local/flann/lib/ -lflann_cuda
g++ -c test_wrapper.C -o test_wrapper.o
gcc -c test.c -o test.o
gcc -O3 test.o test_wrapper.o test_cuda.o -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -l:liblz4.a -L /usr/local/flann/lib/ -lflann_cuda -lstdc++ -o test_cuda