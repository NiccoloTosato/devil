#include<cuda.h>
#include<stdio.h>
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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

#define cudaCheckError() {                                          \
    cudaError_t e=cudaGetLastError();				    \
    if(e!=cudaSuccess) {						\
      printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
      exit(0);								\
    }									\
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

__global__ void expGPU(float *vector) {
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  vector[x]=exp(vector[x]);
}

//    mu_g = (k + y.array()) / (1 + k * w_q.array());
__global__ void line30(float *mu_g,float *y,float *w_q, float k) {
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  mu_g[idx] = (k+y[idx])/(1+k *w_q[idx]);
}


__global__ void diag(float *matrix,float *array1, float* array2,int n) {
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  matrix[idx*n+idx] = array1[idx]*array2[idx];
}


int main() {

  int cols=32;
  int rows=32;


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

  cublasHandle_t handle;
  cublasCreate(&handle);
  ////////////////////////////////////////
  //cublas max test !
  /*
  float value = 99.0;
  int i=3;
  cudaMemcpy(off_device + i, &value, sizeof(float), cudaMemcpyHostToDevice);

  cublasStatus_t status;
  int result=-99;
  status=  cublasIsamax(handle, off.size(),
	       off_device,1,&result);
  printf("Max index %d\n",result);

  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("cuBLAS error\n");
} else {
    printf("cuBLAS function executed successfully\n");
}
  #ifdef DEBUG
  printf("off_device\n");
  printArray<<<1,1,1>>>(off_device,off.size());
  fflush(stdout);
  cudaDeviceSynchronize();
  #endif
  */
  ///////////////////////////////////////
  int iter=0;

    while(iter < 10) {
    //line 29
    //w_q = (-X * mu_beta - off)
    // Perform the operation y := alpha*A*x + beta*y
    float alpha=-1.0;
    float beta=1.0;
    cublasSgemv(handle, CUBLAS_OP_N, rows, cols, &alpha, X_device, cols, mu_beta_device, 1, &beta, w_q_device, 1);

    //w_q = w_q.exp()
    dim3 threadsPerBlock(32);
    dim3 numBlocks_array_cols(cols / threadsPerBlock.x);
    expGPU<<<numBlocks_array_cols,threadsPerBlock, 1>>>(w_q_device);
    // line 30
    // mu_g = (k + y.array()) / (1 + k * w_q.array());
    dim3 numBlocks_array_rows(rows  / threadsPerBlock.x);
    float k=3.0;
    line30<<<numBlocks_array_rows, threadsPerBlock>>>(mu_g_device,y_device,w_q_device,k);
    float* diagonalMatrix;
    CUDA_CHECK(   cudaMalloc((void**)&diagonalMatrix, cols*cols * sizeof(float)) );
    //line 31
    //    Zigma = (X.transpose() * (mu_g.array() * w_q.array()).matrix().asDiagonal() * X).inverse();
    diag<<<numBlocks_array_rows,threadsPerBlock,1>>>(diagonalMatrix,mu_g_device,w_q_device,rows);

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, cols, 1.0, X, cols, diagonalMatrix, cols, 0.0, Zigma_device, cols);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, cols, cols, 1.0, Zigma_device, cols, X, cols, 0.0, diagonalMatrix, cols);

    float* tmp;
    tmp=Zigma_device;
    Zigma_device=diagonalMatrix;
    diagoanlMatrix=tmp;
    inverseMatrix(diagonalMatrix,)
    iter++;
  }
    
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
