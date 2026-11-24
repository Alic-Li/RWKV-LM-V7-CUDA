#pragma once
#include <hip/hip_bf16.h>

using __nv_bfloat16 = __hip_bfloat16;

#define __float2bfloat16_rn __float2bfloat16
