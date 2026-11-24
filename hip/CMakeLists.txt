cmake_minimum_required(VERSION 3.24)
project(rwkv7_hip_ops LANGUAGES HIP)

add_library(rwkv7_hip_ops STATIC
  rwkv7_clampw.hip
  rwkv7_clampw_v3_for_h100.hip
  rwkv7_cmix_bf16_v5.hip
  rwkv7_head_l2wrap_ce_bf16_v4.hip
  rwkv7_tmix_a_gate_bf16.hip
  rwkv7_tmix_kk_pre_bf16_v5.hip
  rwkv7_tmix_lnx_rkvres_xg_bf16_v1.hip
  rwkv7_tmix_mix6_bf16_v5.hip
  rwkv7_tmix_vres_gate_bf16_v1.hip
)
target_include_directories(rwkv7_hip_ops PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(rwkv7_hip_ops PRIVATE _N_=64 _CHUNK_LEN_=16)
set_target_properties(rwkv7_hip_ops PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED YES)
