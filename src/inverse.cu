#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Failed with error code %s at line %d in file %s\n", \
                    cudaGetErrorString(err), __LINE__, __FILE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CUSOLVER(call) \
    do { \
        cusolverStatus_t stat = call; \
        if (stat != CUSOLVER_STATUS_SUCCESS) { \
            fprintf(stderr, "Failed with error code %d at line %d in file %s\n", \
                    stat, __LINE__, __FILE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


void transposeMatrix(float* matrix,int n) {
  int i, j;
  float temp;
  for (i = 0; i < n; i++) {
    for (j = i+1; j < n; j++) {
      temp = matrix[i*n+j];
      matrix[i*n+j] = matrix[j*n+i];
      matrix[j*n+i] = temp;
    }
  }
}
__global__ void printMatrixDevice(float* matrix,int n) 
{
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%f ", matrix[i*n+j]);
        }
        printf("\n");
    }
}

void printMatrix(float* matrix,int n) 
{
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%f ", matrix[i*n+j]);
        }
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
void checkCublasStatus(cublasStatus_t status, const char* functionName) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("%s failed with error code %d\n", functionName, status);
    }
}
void inverseMatrix(float *A,float * A_inv, int n) {
  cusolverDnHandle_t cusolverH = NULL;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
  float *d_A = NULL; // device copy of A
  int *d_Ipiv = NULL; // pivot indices
  int *d_info = NULL; // error info
  int lwork = 0; // size of workspace
  float *d_work = NULL; // device workspace for getrf
  int dev = 0;
  cudaDeviceProp deviceProp;

  CHECK_CUDA(cudaSetDevice(dev));
  CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, dev));
  printf("Device %d: %s\n", dev, deviceProp.name);
  /////////////////////////////////////////////////////////////////////
  //allocate stuff on device
  CHECK_CUDA(cudaMallocManaged(&d_A, sizeof(float)*n*n));
  CHECK_CUDA(cudaMallocManaged(&d_Ipiv, sizeof(int)*n));
  CHECK_CUDA(cudaMallocManaged(&d_info, sizeof(int)));
  //move A to d_A
  CHECK_CUDA(cudaMemcpy(d_A, A, sizeof(float)*n*n, cudaMemcpyHostToDevice));
  
  CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));
  CHECK_CUSOLVER(cusolverDnSgetrf_bufferSize(cusolverH, n, n, d_A, n, &lwork));
  
  CHECK_CUDA(cudaMallocManaged(&d_work, sizeof(float)*lwork));
  CHECK_CUSOLVER(cusolverDnSgetrf(cusolverH, n, n, d_A, n, d_work, d_Ipiv, d_info));
  //printf("LU matrix factorized\n");
  //  printMatrixDevice<<<1,1>>>(d_A,n);
  //  fflush(stdout);
  //  cudaDeviceSynchronize();
  
  if (*d_info != 0 )
    printf("d_info %d ERRORRRRR \n",*d_info);
    CHECK_CUDA(cudaDeviceSynchronize());
    assert(0 == d_info[0]);

    // Copying the LU decomposed matrix back to host
    //CHECK_CUDA(cudaMemcpy((void**)A,(void**) d_A, sizeof(float)*n*n, cudaMemcpyDeviceToHost));
    //transposeMatrix(A,n);
    //printf("A after fact:\n");
    //printMatrix(A,n);
    /*
    */

    float* eye;
    cudaMalloc((void**)&eye, n*n * sizeof(float)) ;
    dim3 blockDim(1,1);
    dim3 gridDim(n, n);  
    initIdentityGPU<<<blockDim, gridDim>>>(eye, n, n, 1.0);
    // Solve the system AX = I
    cusolver_status = cusolverDnSgetrs(
				       cusolverH,
				       CUBLAS_OP_N,
				       n,
				       n,
				       d_A,
				       n,
				       d_Ipiv,
				       eye, // This should be the identity matrix
				       n,
				       d_info);
    //printf("eye  matrix\n");
    //printMatrixDevice<<<1,1>>>(eye,n);
    //fflush(stdout);
    //cudaDeviceSynchronize();
    
    //// check with dgemm ////
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha=1.0;
    float beta=1.0;
    float* matrix_check;
    cudaMalloc((void**)&matrix_check, n*n * sizeof(float)) ;
    CHECK_CUDA(cudaMemcpy((void**)A_inv,(void**) eye, sizeof(float)*n*n, cudaMemcpyDeviceToHost));
    cudaMemset(matrix_check,0,n*n*sizeof(float));
    CHECK_CUDA(cudaMemcpy(d_A, A, sizeof(float)*n*n, cudaMemcpyHostToDevice));
    checkCublasStatus(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, eye, n, d_A, n, &beta, matrix_check, n),"Sgemm");
    printf("Print A_inv*A matrix\n");
    printMatrixDevice<<<1,1>>>(matrix_check,n);
    fflush(stdout);
    cudaDeviceSynchronize();
    
    if (d_A    ) cudaFree(d_A);
    if (d_Ipiv ) cudaFree(d_Ipiv);
    if (d_info ) cudaFree(d_info);
    if (d_work ) cudaFree(d_work);
    if (cusolverH) cusolverDnDestroy(cusolverH);
    cudaDeviceReset();
}

int main(int argc, char*argv[]) {
    int n = 3; // size of matrix
    float A[9] = {1,2,3,4,13,6,17,23,10}; // input matrix
    float A_inv[9] = {1,2,3,4,5,6,7,8,10}; // input matrix

    transposeMatrix(A,n);
    printf("A matrix:\n");
    printMatrix(A,n);
    
    inverseMatrix(A, A_inv, n);
    transposeMatrix(A_inv,n);
    //printf("A inverted\n");
    //printMatrix(A_inv,n);


    return 0;
}
