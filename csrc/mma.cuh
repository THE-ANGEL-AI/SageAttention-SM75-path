/*
 * Adapted from Flashinfer, https://github.com/flashinfer-ai/flashinfer/blob/v0.1.5/include/flashinfer/mma.cuh
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Modifications copyright (c) 2024 by SageAttention team.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace mma{

#if (__CUDACC_VER_MAJOR__ >= 11)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))
#define MMA_F16F16F32_M16N8K16_ENABLED
#define MMA_F16F16F16_M16N8K16_ENABLED
#define MMA_S8S8S32_M16N8K32_ENABLED
#define MMA_S4S4S32_M16N8K64_ENABLED
#endif
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 750))
#define MMA_F16F16F32_M16N8K8_ENABLED
#define MMA_F16F16F16_M16N8K8_ENABLED
#define LDMATRIX_M8N8X2_ENABLED
#define LDMATRIX_M8N8X4_ENABLED
#endif
#endif

#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120400)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 890))
#define MMA_F8F8F32_M16N8K16_ENABLED
#endif
#endif

#if defined(__CUDA_ARCH__)
#define RUNTIME_ASSERT(x) __brkpt()
#else
#include <assert.h>
#define RUNTIME_ASSERT(x) assert(0 && x)
#endif

enum class MMAMode {
  kInit = 0U,
  kInplaceUpdate = 1U,
};

/*!
 * \brief Wrapper of PTX ldmatrix m8n8.x1 instruction, loads 1 register
 *   from shared memory (all 32 threads participate, 1 × .b32 each).
 *   Used for SM75 m8n8k16 MMA where A/B operands need only 1 register.
 * \tparam T data type of the fragment
 * \param R pointer to the fragment (single uint32_t)
 * \param smem_ptr pointer to the shared memory
 */
template <typename T>
__device__ __forceinline__ void ldmatrix_m8n8x1(uint32_t* R, T* smem_ptr) {
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 750))
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
               : "=r"(R[0])
               : "r"(smem_int_ptr));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for ldmatrix instruction");
#endif
}

/*!
 * \brief Wrapper of PTX ldmatrix m8n8.x1 transposed instruction,
 *   loads 1 register from shared memory and transposes the fragment.
 *   Used for V (B operand) in SM75 PV MMA where V is row-major in smem.
 * \tparam T data type of the fragment
 * \param R pointer to the fragment (single uint32_t)
 * \param smem_ptr pointer to the shared memory
 */
template <typename T>
__device__ __forceinline__ void ldmatrix_m8n8x1_trans(uint32_t* R, T* smem_ptr) {
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 750))
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.trans.m8n8.x1.shared.b16 {%0}, [%1];\n"
               : "=r"(R[0])
               : "r"(smem_int_ptr));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for ldmatrix instruction");
#endif
}

/*!
 * \brief Wrapper of PTX ldmatrix m8n8.x2 instruction, loads data from shared memory
 *   to fragment
 * \tparam T data type of the fragment
 * \param R pointer to the fragment
 * \param smem_ptr pointer to the shared memory
 */
template <typename T>
__device__ __forceinline__ void ldmatrix_m8n8x2(uint32_t* R, T* smem_ptr) {
#ifdef LDMATRIX_M8N8X2_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(R[0]), "=r"(R[1])
               : "r"(smem_int_ptr));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for ldmatrix instruction");
#endif
}

/*!
 * \brief Wrapper of PTX ldmatrix m8n8.x4 instruction, loads data from shared memory
 *   to fragment
 * \tparam T data type of the fragment
 * \param R pointer to the fragment
 * \param smem_ptr pointer to the shared memory
 */
template <typename T>
__device__ __forceinline__ void ldmatrix_m8n8x4(uint32_t* R, T* smem_ptr) {
#ifdef LDMATRIX_M8N8X4_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
               : "r"(smem_int_ptr));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for ldmatrix instruction");
#endif
}

/*!
 * \brief Wrapper of PTX ldmatrix m8n8.x4 transposed instruction, loads data from
 *   shared memory to fragment and transposes the fragment
 * \tparam T data type of the fragment
 * \param R pointer to the fragment
 * \param smem_ptr pointer to the shared memory
 */
template <typename T>
__device__ __forceinline__ void ldmatrix_m8n8x4_trans(uint32_t* R, T* smem_ptr) {
#ifdef LDMATRIX_M8N8X4_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.trans.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
               : "r"(smem_int_ptr));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for ldmatrix instruction");
#endif
}

/*!
 * \brief Wrapper of the mma m16n8k16 instruction for row major and column major f16 matrix
 *   multiplication, accumulated in f32.
 * \tparam mma_mode The mode of mma instruction, either kInit or kInplaceUpdate
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n8k16_row_col_f16f16f32(float* C, uint32_t* A,
                                                                     uint32_t* B) {
#ifdef MMA_F16F16F32_M16N8K16_ENABLED
  // ! only support half dtype now
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]),
          "f"(C[2]), "f"(C[3]));
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(0.f), "f"(0.f),
          "f"(0.f), "f"(0.f));
  }
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

/*!
 * \brief Wrapper of the mma m16n16k16 instruction for row major and column major f16 matrix
 *   multiplication, accumulated in f32.
 * \tparam mma_mode The mode of mma instruction, either kInit or kInplaceUpdate
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k16_row_col_f16f16f32(float* C, uint32_t* A,
                                                                     uint32_t* B) {
#ifdef MMA_F16F16F32_M16N8K16_ENABLED
  // ! only support half dtype now
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]),
            "f"(C[2]), "f"(C[3]));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(C[4]), "f"(C[5]),
            "f"(C[6]), "f"(C[7]));
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(0.f), "f"(0.f),
          "f"(0.f), "f"(0.f));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(0.f), "f"(0.f),
          "f"(0.f), "f"(0.f));
  }
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

/*!
 * \brief Wrapper of the mma m16n8k16 instruction for row major and column major f16 matrix
 *   multiplication, accumulated in f16.
 * \tparam mma_mode The mode of mma instruction, either kInit or kInplaceUpdate
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n8k16_row_col_f16f16f16(uint32_t* C, uint32_t* A,
                                                                     uint32_t* B) {
#ifdef MMA_F16F16F16_M16N8K16_ENABLED
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]));
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(0), "r"(0));
  }
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

/*!
 * \brief Wrapper of the mma m16n16k16 instruction for row major and column major f16 matrix
 *   multiplication, accumulated in f16.
 * \tparam mma_mode The mode of mma instruction, either kInit or kInplaceUpdate
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k16_row_col_f16f16f16(uint32_t* C, uint32_t* A,
                                                                     uint32_t* B) {
#ifdef MMA_F16F16F16_M16N8K16_ENABLED
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(C[2]), "r"(C[3]));
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(0), "r"(0));
  }
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

/*!
 * \brief Wrapper of the mma m16n8k32 instruction for row major and column major int8 matrix
 *   multiplication, accumulated in int32.
 * \tparam mma_mode The mode of mma instruction, either kInit or kInplaceUpdate
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n8k32_row_col_s8s8s32(int32_t* C, uint32_t* A,
                                                                   uint32_t* B) {
#ifdef MMA_S8S8S32_M16N8K32_ENABLED
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]),
          "r"(C[2]), "r"(C[3]));
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(0), "r"(0),
          "r"(0), "r"(0));
  }
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

/*!
 * \brief Wrapper of the mma m16n16k32 instruction for row major and column major int8 matrix
 *   multiplication, accumulated in int32.
 * \tparam mma_mode The mode of mma instruction, either kInit or kInplaceUpdate
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k32_row_col_s8s8s32(int32_t* C, uint32_t* A,
                                                                   uint32_t* B) {
#ifdef MMA_S8S8S32_M16N8K32_ENABLED
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]),
          "r"(C[2]), "r"(C[3]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[4]), "=r"(C[5]), "=r"(C[6]), "=r"(C[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(C[4]), "r"(C[5]),
          "r"(C[6]), "r"(C[7]));
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(0), "r"(0),
          "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[4]), "=r"(C[5]), "=r"(C[6]), "=r"(C[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(0), "r"(0),
          "r"(0), "r"(0));
  }
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

/*!
 * \brief Wrapper of the mma m16n8k32 instruction for row major and column major int4 matrix
 *   multiplication, accumulated in int32.
 * \tparam mma_mode The mode of mma instruction, either kInit or kInplaceUpdate
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n8k64_row_col_s4s4s32(int32_t* C, uint32_t* A,
                                                                   uint32_t* B) {
#ifdef MMA_S4S4S32_M16N8K64_ENABLED
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]),
          "r"(C[2]), "r"(C[3]));
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(0), "r"(0),
          "r"(0), "r"(0));
  }
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

/*!
 * \brief Wrapper of the mma m16n16k64 instruction for row major and column major int4 matrix
 *   multiplication, accumulated in int32.
 * \tparam mma_mode The mode of mma instruction, either kInit or kInplaceUpdate
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k64_row_col_s4s4s32(int32_t* C, uint32_t* A,
                                                                   uint32_t* B) {
#ifdef MMA_S4S4S32_M16N8K64_ENABLED
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]),
          "r"(C[2]), "r"(C[3]));
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[4]), "=r"(C[5]), "=r"(C[6]), "=r"(C[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(C[4]), "r"(C[5]),
          "r"(C[6]), "r"(C[7]));
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(0), "r"(0),
          "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[4]), "=r"(C[5]), "=r"(C[6]), "=r"(C[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(0), "r"(0),
          "r"(0), "r"(0));
  }
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

/*!
 * \brief Wrapper of the mma m16n8k32 instruction for row major and column major fp8 e4m3 matrix
 *   multiplication, accumulated in fp32.
 * \tparam mma_mode The mode of mma instruction, either kInit or kInplaceUpdate
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n8k32_row_col_f8f8f32(float* C, uint32_t* A,
                                                                   uint32_t* B) {
#ifdef MMA_F8F8F32_M16N8K16_ENABLED
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]),
          "f"(C[2]), "f"(C[3]));
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(0.f), "f"(0.f),
          "f"(0.f), "f"(0.f));
  }
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

/*!
 * \brief Wrapper of the mma m16n16k32 instruction for row major and column major fp8 matrix
 *   multiplication, accumulated in fp32.
 * \tparam mma_mode The mode of mma instruction, either kInit or kInplaceUpdate
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k32_row_col_f8f8f32(float* C, uint32_t* A,
                                                                   uint32_t* B) {
#ifdef MMA_F8F8F32_M16N8K16_ENABLED
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]),
          "f"(C[2]), "f"(C[3]));
    
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(C[4]), "f"(C[5]),
          "f"(C[6]), "f"(C[7]));
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(0.f), "f"(0.f),
          "f"(0.f), "f"(0.f));

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(0.f), "f"(0.f),
          "f"(0.f), "f"(0.f));
  }
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

/*!
 * \brief Use mma instructions to compute rowsum.
 */
__device__ __forceinline__ void rowsum_f16f16f32(float* d, uint32_t* s) {
#ifdef MMA_F16F16F32_M16N8K16_ENABLED
  asm volatile(
      "{\n"
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,  _,  %1,  _},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  0.,  %9,  0.};\n"
      "}\n"
      : "=f"(d[0]), "=f"(d[1])
      : "r"(s[0]), "r"(s[1]), "r"(s[2]), "r"(s[3]), "r"(1006648320), // 1006648320 packs two 1.0f in half precision
        "r"(1006648320), "f"(d[0]), "f"(d[1]));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

/*!
 * \brief Use mma instructions to compute rowsum.
 */
__device__ __forceinline__ void rowsum_f8f8f32(float* d, uint32_t* s) {
#ifdef MMA_F8F8F32_M16N8K16_ENABLED
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,  _,  %1,  _},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  0.,  %9,  0.};\n"
      : "=f"(d[0]), "=f"(d[1])
      : "r"(s[0]), "r"(s[1]), "r"(s[2]), "r"(s[3]), "r"(943208504), "r"(943208504), // 943208504 packs four 1.0f in e4m3
        "f"(d[0]), "f"(d[1]));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

// ===== SM75 (Turing) MMA wrappers =====
// All SM75 wrappers require __CUDA_ARCH__ >= 750.
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 750))
#define MMA_SM75_ENABLED
#endif

// mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32
// Valid for SM75+ int8 tensor cores:
//   A = 1 × uint32  (4 × s8 packed into 1 × b32)
//   B = 1 × uint32  (4 × s8 packed into 1 × b32)
//   C/D = 2 × int32 (2 × s32 per thread, each holds 4 elements of the 8×8 output)
// NOTE: m8n8k32 does NOT exist in PTX ISA. K=32 variants use m16n8k32 for SM80+.
// NOTE: m16n8k8 does NOT exist in PTX ISA for SM75 — only SM80+.
// NOTE: PTX requires D and C operands to be non-overlapping registers.
// We use earlyclobber ("=&") to enforce disjoint output/input register allocation.
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m8n8k16_row_col_s8s8s32(int32_t* C, uint32_t* A, uint32_t* B) {
#ifdef MMA_SM75_ENABLED
  int32_t C_out[2];
  if constexpr (mma_mode == MMAMode::kInplaceUpdate) {
    asm volatile(
        "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
        "{%0, %1},"
        "{%2},"
        "{%3},"
        "{%4, %5};\n"
        : "=&r"(C_out[0]), "=&r"(C_out[1])
        : "r"(A[0]),
          "r"(B[0]),
          "r"(C[0]), "r"(C[1]));
  } else {
    asm volatile(
        "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
        "{%0, %1},"
        "{%2},"
        "{%3},"
        "{%4, %5};\n"
        : "=&r"(C_out[0]), "=&r"(C_out[1])
        : "r"(A[0]),
          "r"(B[0]),
          "r"(0), "r"(0));
  }
  C[0] = C_out[0];
  C[1] = C_out[1];
#else
  RUNTIME_ASSERT("mma_sync_m8n8k16_row_col_s8s8s32 requires SM75+ (Turing)");
#endif
}

// mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32
// Valid for SM75+ (Turing) fp16 tensor cores:
//   A = 1 × uint32 (2 × f16 packed into 1 × b32)
//   B = 1 × uint32 (2 × f16 packed into 1 × b32)
//   C/D = 2 × float (2 × f32 per thread, each holds 4 elements of the 8×8 output)
// NOTE: m8n8k4 is the ONLY fp16 MMA shape valid on SM75 (Turing).
//       m8n8k16 fp16 requires SM80+ (Ampere).
// NOTE: PTX requires D and C operands to be non-overlapping registers.
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m8n8k4_row_col_f16f16f32(float* C, uint32_t* A, uint32_t* B) {
#ifdef MMA_SM75_ENABLED
  float C_out[2];
  if constexpr (mma_mode == MMAMode::kInplaceUpdate) {
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0, %1},"
        "{%2},"
        "{%3},"
        "{%4, %5};\n"
        : "=&f"(C_out[0]), "=&f"(C_out[1])
        : "r"(A[0]),
          "r"(B[0]),
          "f"(C[0]), "f"(C[1]));
  } else {
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0, %1},"
        "{%2},"
        "{%3},"
        "{%4, %5};\n"
        : "=&f"(C_out[0]), "=&f"(C_out[1])
        : "r"(A[0]),
          "r"(B[0]),
          "f"(0.0f), "f"(0.0f));
  }
  C[0] = C_out[0];
  C[1] = C_out[1];
#else
  RUNTIME_ASSERT("mma_sync_m8n8k4_row_col_f16f16f32 requires SM75+ (Turing)");
#endif
}
// ===== END SM75 wrappers =====

} // namespace mma
