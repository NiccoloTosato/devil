#include<cuda.h>
#include<stdio.h>
#include <Eigen/Dense>

#define  BLOCK_DIM_X 32
#define  BLOCK_DIM_Y 32

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__ void printArray(float* array,int size) {
  for(int i=0;i<size;++i)
    printf("%2.2f ",array[i]);
  printf("\n");
}


__global__ void printMatrix(float* array,int rows,int cols) {
  for(int i=0;i<rows;++i) {
    for(int j=0;j<cols;++j)
      printf("%2.2f ",array[i*cols+j]);
    printf("\n");
  }
}

__global__ void initIdentityGPU(float *Matrix, int rows, int cols,float alpha) {
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
  if(y < rows && x < cols) {
    if(x == y)
      Matrix[y*cols+x] = 1*alpha;
    else
      Matrix[y*cols+x] = 0;
    }
}

int main() {

  int cols=10;
  int rows=10;


  //block stuff, to be redefined ! 
  dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);  
  dim3 gridDim((cols + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (rows + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);


  float* delta_device;  
  CUDA_CHECK( cudaMalloc((void**)&delta_device, cols * sizeof(float)) );
  CUDA_CHECK( cudaMemset(delta_device,0,cols*sizeof(float)) );

  #ifdef DEBUG
  printf("delta_device\n");
  printArray<<<1,1,1>>>(delta_device,cols);
  fflush(stdout);
  cudaDeviceSynchronize();
  #endif

  float* inv_sigma_beta_const_device;
  CUDA_CHECK( cudaMalloc((void**)&inv_sigma_beta_const_device, cols*cols * sizeof(float)) );
  initIdentityGPU<<<gridDim, blockDim>>>(inv_sigma_beta_const_device, cols, cols, 0.01);

  float* Zigma_device;
  CUDA_CHECK( cudaMalloc((void**)&Zigma_device, cols*cols * sizeof(float)) );
  initIdentityGPU<<<gridDim, blockDim>>>(Zigma_device, cols, cols, 1.0);

  #ifdef DEBUG
  printf("inv_sigma_beta_const\n");
  printMatrix<<<1,1,1>>>(inv_sigma_beta_const_device,cols,cols);
  fflush(stdout);
  cudaDeviceSynchronize();
  #endif

  float* mu_g_device;
  CUDA_CHECK( cudaMalloc((void**)&mu_g_device, rows * sizeof(float)) );
  CUDA_CHECK( cudaMemset(mu_g_device,0,rows*sizeof(float)) );

  float* w_q_device;
  CUDA_CHECK( cudaMalloc((void**)&w_q_device, rows * sizeof(float)) );
  CUDA_CHECK( cudaMemset(w_q_device,0,rows*sizeof(float)) );


  //dummy input:, this stuff will be moved to the function arg soon
  Eigen::MatrixXf X(rows,cols);
  Eigen::VectorXf y(rows);
  Eigen::VectorXf mu_beta(cols);
  Eigen::VectorXf off(rows);

  float* X_device;
  CUDA_CHECK(   cudaMalloc((void**)&X_device, X.size() * sizeof(float)) );
  CUDA_CHECK(   cudaMemcpy(X_device, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice) );

  float* y_device;
  CUDA_CHECK(   cudaMalloc((void**)&y_device, y.size() * sizeof(float)) );
  CUDA_CHECK(   cudaMemcpy(y_device, y.data(), y.size() * sizeof(float), cudaMemcpyHostToDevice) );

  float* mu_beta_device;
  CUDA_CHECK(   cudaMalloc((void**)&mu_beta_device, mu_beta.size() * sizeof(float)) );
  CUDA_CHECK(   cudaMemcpy(y_device, y.data(), y.size() * sizeof(float), cudaMemcpyHostToDevice) );

  float* off_device;
  CUDA_CHECK(   cudaMalloc((void**)&off_device, off.size() * sizeof(float)) );
  CUDA_CHECK(   cudaMemcpy(off_device, off.data(), off.size() * sizeof(float), cudaMemcpyHostToDevice) );
  int iter=0;
  while(iter < 10) {

  }
    //  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_col,  n_fix, N , &alpha, buffer_device, n_col, A_device, N, &beta, C_device+offset, N);
    
  CUDA_CHECK( cudaFree(delta_device) );
  CUDA_CHECK( cudaFree(inv_sigma_beta_const_device) );
  CUDA_CHECK( cudaFree(Zigma_device) );
  CUDA_CHECK( cudaFree(mu_g_device) );
  CUDA_CHECK( cudaFree(w_q_device) );
  CUDA_CHECK( cudaFree(X_device) );
  CUDA_CHECK( cudaFree(y_device) );
  CUDA_CHECK( cudaFree(mu_beta_device) );
  CUDA_CHECK( cudaFree(off_device) );
  
}
