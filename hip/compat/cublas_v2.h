#pragma once
#include <hipblas/hipblas.h>

using cudaDataType_t = hipDataType;
using cublasHandle_t = hipblasHandle_t;
using cublasStatus_t = hipblasStatus_t;

#define CUDA_R_16BF HIP_R_16BF
#define CUDA_R_32F HIP_R_32F
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasGemmEx hipblasGemmEx
#define cublasGetStatusString hipblasStatusToString
#define cublasSetStream hipblasSetStream
