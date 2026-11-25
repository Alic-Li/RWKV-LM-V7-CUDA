#pragma once
#include <hipsolver/hipsolver.h>

using cusolverDnHandle_t = hipsolverHandle_t;
using cusolverStatus_t = hipsolverStatus_t;

#define CUSOLVER_STATUS_SUCCESS HIPSOLVER_STATUS_SUCCESS
#define cusolverDnCreate hipsolverDnCreate
#define cusolverDnDestroy hipsolverDnDestroy
#define cusolverDnSgeqrf hipsolverDnSgeqrf
#define cusolverDnSgeqrf_bufferSize hipsolverDnSgeqrf_bufferSize
#define cusolverDnSorgqr hipsolverDnSorgqr
#define cusolverDnSorgqr_bufferSize hipsolverDnSorgqr_bufferSize
#define cusolverDnSetStream hipsolverSetStream
