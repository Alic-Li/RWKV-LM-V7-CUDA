cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(rwkv7_pretrain LANGUAGES CXX CUDA)

find_package(CUDAToolkit 12 REQUIRED)
find_package(LibArchive REQUIRED)
option(RWKV_NCCL "Enable NCCL data parallel training when available" ON)

add_library(rwkv_pth STATIC
  pretrain/weights.cpp pretrain/compat/pth_archive.cpp pretrain/compat/pth_tensor.cpp)
target_compile_features(rwkv_pth PUBLIC cxx_std_17)
target_include_directories(rwkv_pth PUBLIC pretrain/compat)
target_link_libraries(rwkv_pth PRIVATE LibArchive::LibArchive)

add_executable(rwkv7_init pretrain/init.cu)
target_link_libraries(rwkv7_init PRIVATE rwkv_pth CUDA::cusolver)
add_executable(rwkv7_train pretrain/train.cpp pretrain/dataset.cpp pretrain/model.cu
  pretrain/compat/jsonl.cpp
  cuda/rwkv7_clampw.cu cuda/rwkv7_clampw_v3_for_h100.cu
  cuda/rwkv7_cmix_bf16_v5.cu cuda/rwkv7_head_l2wrap_ce_bf16_v4.cu
  cuda/rwkv7_tmix_a_gate_bf16.cu cuda/rwkv7_tmix_kk_pre_bf16_v5.cu
  cuda/rwkv7_tmix_lnx_rkvres_xg_bf16_v1.cu cuda/rwkv7_tmix_mix6_bf16_v5.cu
  cuda/rwkv7_tmix_vres_gate_bf16_v1.cu)
target_include_directories(rwkv7_train PRIVATE cuda)
target_compile_definitions(rwkv7_train PRIVATE _N_=64 _CHUNK_LEN_=16)
target_compile_options(rwkv7_train PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
target_link_libraries(rwkv7_train PRIVATE rwkv_pth CUDA::cudart CUDA::cublas)
if(RWKV_NCCL)
  find_path(NCCL_INCLUDE_DIR nccl.h)
  find_library(NCCL_LIBRARY nccl)
  if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
    target_include_directories(rwkv7_train PRIVATE "${NCCL_INCLUDE_DIR}")
    target_link_libraries(rwkv7_train PRIVATE "${NCCL_LIBRARY}")
    target_compile_definitions(rwkv7_train PRIVATE RWKV_HAVE_NCCL)
  else()
    message(STATUS "NCCL unavailable; building single-GPU training")
  endif()
endif()
set_target_properties(rwkv7_init rwkv7_train PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
