#pragma once
#include <rocblas/rocblas.h>

using cudaDataType_t = rocblas_datatype;
using cublasHandle_t = rocblas_handle;
using cublasStatus_t = rocblas_status;
#define CUDA_R_16BF rocblas_datatype_bf16_r
#define CUDA_R_32F rocblas_datatype_f32_r
#define CUBLAS_OP_N rocblas_operation_none
#define CUBLAS_OP_T rocblas_operation_transpose
#define CUBLAS_STATUS_SUCCESS rocblas_status_success
#define cublasCreate rocblas_create_handle
#define cublasDestroy rocblas_destroy_handle
#define cublasGetStatusString rocblas_status_to_string
#define cublasSetStream rocblas_set_stream
#define cublasSetWorkspace rocblas_set_workspace
