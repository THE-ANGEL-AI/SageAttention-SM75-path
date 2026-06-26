/*
 * Copyright (c) 2024 by SageAttention team.
 * (SM75 Kernel Implementation)
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

#ifdef __CUDACC__

#include "../utils.cuh"
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h> // C10_CUDA_KERNEL_LAUNCH_CHECK

#include "../mma.cuh" // Contains SM75 MMA wrappers now
#include "../math.cuh"
#include "../dispatch_utils.h"
#include "attn_utils.cuh" // Contains shared enums and helpers
#include "../reduction_utils.cuh" // vllm::warpReduceSum, warpReduceMax

// Define SM75 specific constants (Adjust based on tuning)
#define PACK_SIZE_INT8 4  // Loading 4 int8s into a uint32_t
#define PACK_SIZE_FP16 2  // Loading 2 halfs into a uint32_t

// SM75 MMA Shapes
#define MMA_QK_M_SM75 8
#define MMA_QK_N_SM75 8
#define MMA_QK_K_SM75 32 // INT8 MMA K dim (m8n8k32 for Turing)

#define MMA_SV_M_SM75 16
#define MMA_SV_N_SM75 8
#define MMA_SV_K_SM75 8 // FP16 MMA K dim

// --- SM75 Kernel Implementation ---
template< uint32_t CTA_Q, uint32_t CTA_K, uint32_t WARP_Q, uint32_t WARP_K, uint32_t head_dim,
          QuantGranularity Q_GRAN, QuantGranularity K_GRAN,
          typename DTypeOut, MaskMode mask_mode, bool return_lse>
__global__ void qk_int8_sv_f16_accum_f32_attn_kernel_sm75(
                    int8_t *__restrict__ Q, int8_t *__restrict__ K, half *__restrict__ V,
                    DTypeOut *__restrict__ O, float *__restrict__ Lse,
                    float *__restrict__ Q_scale, float *__restrict__ K_scale,
                    const uint32_t qo_len, const uint32_t kv_len, const uint32_t num_kv_groups,
                    const uint32_t stride_bz_q, const uint32_t stride_seq_q, const uint32_t stride_h_q,
                    const uint32_t stride_bz_k, const uint32_t stride_seq_k, const uint32_t stride_h_k,
                    const uint32_t stride_bz_v, const uint32_t stride_seq_v, const uint32_t stride_h_v,
                    const uint32_t stride_bz_o, const uint32_t stride_seq_o, const uint32_t stride_h_o,
                    float sm_scale)
{
    // SM75 supports FP16 only for V/O with this kernel path
    static_assert(std::is_same<DTypeOut, half>::value, "SM75 kernel only supports FP16 output.");

    // --- Thread/Block Indexing ---
    const uint32_t lane_id = threadIdx.x % 32; // Lane index within the warp (0-31)
    const uint32_t warp_id_in_block = threadIdx.x / 32; // Warp index within the CTA
    const uint32_t num_warps_per_block = blockDim.x / 32;

    // Divide warps between Q and K dimensions
    // Example: Assign warps round-robin or block-wise. Let's try block-wise.
    const uint32_t warps_per_cta_q = CTA_Q / WARP_Q;
    const uint32_t warps_per_cta_k = CTA_K / WARP_K;
    // assert(num_warps_per_block == warps_per_cta_q * warps_per_cta_k); // Ensure blockDim matches geometry

    const uint32_t warp_idx_q = warp_id_in_block / warps_per_cta_k; // Warp's Q-dimension index
    const uint32_t warp_idx_k = warp_id_in_block % warps_per_cta_k; // Warp's K-dimension index

    const uint32_t batch_id = blockIdx.z;
    const uint32_t bx = blockIdx.x; // Block index along Q-dimension tiles
    const uint32_t head_id = blockIdx.y;
    const uint32_t num_qo_heads = gridDim.y;
    const uint32_t kv_head_id = head_id / num_kv_groups;

    // --- Shared Memory Allocation ---
    // Use simple row-major layout. Add padding to reduce bank conflicts.
    // Padding amount (e.g., 16 bytes = 8 halfs = 16 int8s)
    // Adjust padding based on profiling if needed.
    constexpr uint32_t SHMEM_PADDING_BYTES = 16;
    constexpr uint32_t HEAD_DIM_PADDED_INT8 = head_dim + (SHMEM_PADDING_BYTES / sizeof(int8_t));
    constexpr uint32_t HEAD_DIM_PADDED_FP16 = head_dim + (SHMEM_PADDING_BYTES / sizeof(half));

    extern __shared__ int8_t smem_storage[];
    int8_t* smem_Q = smem_storage;
    int8_t* smem_K = smem_Q + CTA_Q * HEAD_DIM_PADDED_INT8;
    half*   smem_V = reinterpret_cast<half*>(smem_K + CTA_K * HEAD_DIM_PADDED_INT8);
    // Optional: Allocate shared memory for output tile if needed for coalescing stores
    // half*   smem_O = reinterpret_cast<half*>(smem_V + CTA_K * HEAD_DIM_PADDED_FP16);

    // --- Register Allocation ---
    // QK Accumulator (INT8 MMA -> INT32)
    // Each thread handles a small part of the M*N tile. Size depends on MMA shape.
    // Example: m8n8k4 => each thread needs 2 int32 accumulators? Need to check PTX guide. Assume 2.
    constexpr int NUM_QK_ACCUM = 2; // Placeholder
    int32_t RS_accum[NUM_QK_ACCUM];

    // PV Accumulator (FP16 MMA -> FP32)
    // Example: m16n8k8 => each thread needs 4 FP32 accumulators.
    constexpr int NUM_PV_ACCUM = 4;
    float RO_accum[NUM_PV_ACCUM];

    // Softmax state (per row handled by a group of threads, typically a warp)
    // Each warp handles WARP_Q rows. Each thread handles WARP_Q / warp_size_rows rows?
    // Let's assume each thread handles a fraction of the rows within its warp-Q-tile.
    // Simplification: Each thread tracks m/l for the rows it calculates within the MMA tile.
    // E.g., for m16n8k8 PV MMA, each thread calculates 4 output elements.
    float m_i[NUM_PV_ACCUM]; // Max per output element row fragment
    float l_i[NUM_PV_ACCUM]; // Sum per output element row fragment

    // --- Initialization ---
    #pragma unroll
    for (int i = 0; i < NUM_PV_ACCUM; ++i) {
        RO_accum[i] = 0.0f;
        m_i[i] = -INFINITY; // Use INFINITY macro or a large negative number
        l_i[i] = 0.0f;      // Start sum at 0, will be exp(m_i - m_ij) + ...
    }

    // --- Load Q tile into Shared Memory ---
    // Loop over columns (head_dim) and rows (CTA_Q) assigned to this block
    // Each thread loads multiple elements
    const uint32_t q_start_row_block = bx * CTA_Q;
    #pragma unroll
    for (int i = threadIdx.x; i < CTA_Q * head_dim; i += blockDim.x) {
        uint32_t q_row_local = i / head_dim;
        uint32_t q_col = i % head_dim;
        uint32_t q_row_global = q_start_row_block + q_row_local;

        int8_t q_val = 0;
        if (q_row_global < qo_len) {
            // Calculate global memory address for Q
            uint32_t q_offset = batch_id * stride_bz_q + head_id * stride_h_q + q_row_global * stride_seq_q + q_col;
            q_val = Q[q_offset];
        }
        // Write to shared memory with padding
        smem_Q[q_row_local * HEAD_DIM_PADDED_INT8 + q_col] = q_val;
    }
    __syncthreads(); // Wait for Q to be loaded

    // --- Prepare Scale Factors ---
    sm_scale *= math::log2e; // Use log2 for exp2 intrinsic
    float q_scale_val;
    // Load Q scale factor (depends on Q_GRAN)
    if constexpr (Q_GRAN == QuantGranularity::kPerWarp) {
         uint32_t num_warp_block_q = gridDim.x * warps_per_cta_q;
         uint32_t q_scale_idx = batch_id * num_qo_heads * num_warp_block_q + head_id * num_warp_block_q + bx * warps_per_cta_q + warp_idx_q;
         q_scale_val = Q_scale[q_scale_idx];
    } else { // Fallback to per-block logic if per-thread is too complex
         uint32_t num_block_q = gridDim.x;
         uint32_t q_scale_idx = batch_id * num_qo_heads * num_block_q + head_id * num_block_q + bx;
         q_scale_val = Q_scale[q_scale_idx];
    }
    // K scale is loaded inside the loop

    // --- Main Loop over K/V Tiles ---
    const uint32_t num_k_tiles = div_ceil(kv_len, CTA_K);
    const uint32_t k_boundary_check_iteration = num_k_tiles -1; // Iteration where boundary K check is needed
    const uint32_t k_boundary_len = kv_len % CTA_K == 0 ? CTA_K : kv_len % CTA_K; // Length of last K block

    for (uint32_t k_tile_idx = 0; k_tile_idx < num_k_tiles; ++k_tile_idx) {
        const uint32_t k_start_row_block = k_tile_idx * CTA_K;
        const bool is_boundary_k_iter = (k_tile_idx == k_boundary_check_iteration);
        const uint32_t current_k_block_len = is_boundary_k_iter ? k_boundary_len : CTA_K;

        // Load K & V tiles into Shared Memory for the current iteration
        #pragma unroll
        for (int i = threadIdx.x; i < CTA_K * head_dim; i += blockDim.x) {
            uint32_t k_row_local = i / head_dim;
            uint32_t k_col = i % head_dim;
            uint32_t k_row_global = k_start_row_block + k_row_local;

            int8_t k_val = 0;
            half v_val = __float2half(0.0f);
            if (k_row_global < kv_len) { // Check global boundary
                // Calculate global memory addresses
                uint32_t k_offset = batch_id * stride_bz_k + kv_head_id * stride_h_k + k_row_global * stride_seq_k + k_col;
                uint32_t v_offset = batch_id * stride_bz_v + kv_head_id * stride_h_v + k_row_global * stride_seq_v + k_col;
                k_val = K[k_offset];
                v_val = V[v_offset];
            }
             // Write to shared memory with padding
            smem_K[k_row_local * HEAD_DIM_PADDED_INT8 + k_col] = k_val;
            smem_V[k_row_local * HEAD_DIM_PADDED_FP16 + k_col] = v_val;
        }
        __syncthreads(); // Wait for K, V loading

        // Load K scale factor for this block/warp
        float k_scale_val;
        if constexpr (K_GRAN == QuantGranularity::kPerWarp) {
             uint32_t num_warp_block_k = div_ceil(kv_len, CTA_K) * warps_per_cta_k;
             uint32_t k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * num_warp_block_k + kv_head_id * num_warp_block_k + k_tile_idx * warps_per_cta_k + warp_idx_k;
             k_scale_val = K_scale[k_scale_idx];
        } else { // Fallback to per-block
             uint32_t num_block_k = div_ceil(kv_len, CTA_K);
             uint32_t k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * num_block_k + kv_head_id * num_block_k + k_tile_idx;
             k_scale_val = K_scale[k_scale_idx];
        }
        float current_dequant_scale = q_scale_val * k_scale_val;

        // --- QK^T Computation (using INT8 MMA) ---
        // Each warp computes a WARP_Q x WARP_K tile of S = QK^T
        uint32_t q_start_warp = warp_idx_q * WARP_Q;
        uint32_t k_start_warp = warp_idx_k * WARP_K;

        // Loop structure depends heavily on MMA shape and tile assignment per thread
        // This part requires careful mapping of threads to MMA inputs/outputs.
        // Placeholder for the complex loop structure:
        #pragma unroll
        for(int mq = 0; mq < WARP_Q / MMA_QK_M_SM75; ++mq) { // Iterate over M dimension within warp tile
             #pragma unroll
             for(int nk = 0; nk < WARP_K / MMA_QK_N_SM75; ++nk) { // Iterate over N dimension within warp tile
                // Reset QK accumulators for this S tile fragment
                #pragma unroll
                for(int acc_idx=0; acc_idx < NUM_QK_ACCUM; ++acc_idx) RS_accum[acc_idx] = 0;

                #pragma unroll
                for(int hk = 0; hk < head_dim / MMA_QK_K_SM75; ++hk) { // Iterate over K dimension
                    // 1. Load Q fragment (mA x kA) from smem_Q into registers (Packed into uint32_t)
                    //    Indices: q_start_warp + mq*MMA_QK_M_SM75 + ... (thread-specific offset)
                    //             hk*MMA_QK_K_SM75 + ... (thread-specific offset)
                    uint32_t q_frag_reg; // Example register holding packed Q data
                    // ... loading logic ...


                    // 2. Load K fragment (kB x nB) from smem_K into registers (Packed into uint32_t)
                    //    Indices: k_start_warp + nk*MMA_QK_N_SM75 + ... (thread-specific offset)
                    //             hk*MMA_QK_K_SM75 + ... (thread-specific offset)
                    uint32_t k_frag_reg; // Example register holding packed K data
                    // ... loading logic ...

                    // 3. Perform MMA (m8n8k4)
                    //    mma.m8n8k4(RS_accum, q_frag_reg, k_frag_reg) // Conceptual call
                    mma::mma_sync_m8n8k32_row_col_s8s8s32<mma::MMAMode::kInplaceUpdate>(RS_accum, q_frag_reg, k_frag_reg);

                } // End K dimension loop

                // --- Intermediate Processing (Softmax Prep) ---
                // Process the accumulated RS_accum (int32) for the S tile fragment (mA x nB)
                // This involves: Convert to float, apply scale, masking, find max, exp2, sum.
                // The result (P fragment) should be stored in registers as FP16 (packed).

                float s_frag_f32[NUM_QK_ACCUM * 2]; // Example: 2 floats per int32 accumulator? Check PTX
                uint32_t p_frag_reg[NUM_QK_ACCUM];  // Example: Resulting P fragment (FP16 packed)

                // Convert int32 accumulators to float32
                #pragma unroll
                for(int acc_idx=0; acc_idx < NUM_QK_ACCUM; ++acc_idx) {
                    // Example conversion (may need PTX intrinsics or careful casting)
                    // s_frag_f32[acc_idx*2] = float(RS_accum[acc_idx] & 0xFFFF); // Hypothetical split
                    // s_frag_f32[acc_idx*2+1] = float(RS_accum[acc_idx] >> 16);
                    s_frag_f32[acc_idx] = __int2float_rz(RS_accum[acc_idx]); // More likely cast needed
                 }


                // Apply scale
                #pragma unroll
                for(int i=0; i < NUM_QK_ACCUM * 2; ++i) { // Adjust loop bound based on actual float count
                    s_frag_f32[i] *= current_dequant_scale;
                }

                // Apply Masking (Causal & Boundary)
                 uint32_t s_tile_start_row = q_start_row_block + q_start_warp + mq * MMA_QK_M_SM75;
                 uint32_t s_tile_start_col = k_start_row_block + k_start_warp + nk * MMA_QK_N_SM75;
                 // ... apply mask logic to s_frag_f32 based on global row/col indices and thread position ...
                 #pragma unroll
                 for (int row_idx_local = 0; row_idx_local < MMA_QK_M_SM75; ++row_idx_local) { // Iterate through elements this thread calculates
                     #pragma unroll
                     for (int col_idx_local = 0; col_idx_local < MMA_QK_N_SM75; ++col_idx_local) {
                         uint32_t global_q_idx = s_tile_start_row + row_idx_local; // Simplified - needs thread mapping
                         uint32_t global_k_idx = s_tile_start_col + col_idx_local; // Simplified - needs thread mapping
                         bool is_masked = false;
                         if (global_q_idx < qo_len) { // Check Q boundary first
                              if (is_boundary_k_iter && global_k_idx >= kv_len) {
                                  is_masked = true; // K boundary check
                              } else if (mask_mode == MaskMode::kCausal && global_k_idx > global_q_idx) {
                                  is_masked = true; // Causal check
                              }
                         } else {
                             is_masked = true; // Q boundary check (mask out rows entirely)
                         }

                         if (is_masked) {
                             // Apply masking to the corresponding element in s_frag_f32
                             // Example: s_frag_f32[...] = -INFINITY;
                         }
                     }
                 }


                // Update m_i, l_i, acc (Softmax Calculation) - Operates Per-Row
                // This needs to map the S fragment elements to the correct m_i/l_i/acc registers
                // ... complex softmax update logic adapted from SM80, using s_frag_f32 ...
                // 1. Find max_k(S_ij) for this fragment -> m_ij_frag
                // 2. Update global max m_i = max(m_i, m_ij_frag) across threads in the row (warp shuffle)
                // 3. Calculate P_ij = exp2(S_ij * sm_scale - m_i)
                // 4. Calculate sum_k(P_ij) -> l_ij_frag
                // 5. Update global sum l_i = l_i * exp2(old_m_i - m_i) + l_ij_frag (warp shuffle)
                // 6. Scale existing acc = acc * exp2(old_m_i - m_i)

                // Pack P fragment into FP16 registers (p_frag_reg)
                #pragma unroll
                for(int i=0; i < NUM_QK_ACCUM; ++i) { // Adjust loop bound based on actual float count
                     half2 p_half2 = __float22half2_rn(make_float2(s_frag_f32[i*2], s_frag_f32[i*2+1])); // Example
                     // Pack half2 into uint32_t
                     // p_frag_reg[i] = __half2_as_ushort(p_half2.x) | (__half2_as_ushort(p_half2.y) << 16); // Check packing order
                     ((half2*)&p_frag_reg[i])[0] = p_half2;
                }

                // --- PV Computation (using FP16 MMA) ---
                 #pragma unroll
                 for(int hk = 0; hk < head_dim / MMA_SV_K_SM75; ++hk) { // Iterate over K dim for PV
                     // Load V fragment (kA x nB) from smem_V into registers (Packed into uint32_t)
                     // Indices: k_start_warp + nk*MMA_SV_N_SM75 + ...
                     //          hk*MMA_SV_K_SM75 + ...
                     uint32_t v_frag_reg; // Example: For m16n8k8, B operand is k8 x n8
                     // ... loading logic ...

                     // Perform MMA (m16n8k8)
                     // mma.m16n8k8(RO_accum, p_frag_reg, v_frag_reg) // Conceptual
                     // Note: p_frag_reg needs to match the shape expectation (mA x kA) for m16n8k8
                     mma::mma_sync_m16n8k8_row_col_f16f16f32<mma::MMAMode::kInplaceUpdate>(RO_accum, p_frag_reg, &v_frag_reg); // Example call

                 } // End K dimension loop (PV)

             } // End N dim loop (S tile)
         } // End M dim loop (S tile)
        __syncthreads(); // Sync after finishing work with current K/V tile before loading next
    } // End K tile loop

    // --- Final Normalization & Output ---
    // Normalize RO_accum using final l_i
    #pragma unroll
    for(int i=0; i < NUM_PV_ACCUM; ++i) {
        // Need to aggregate l_i across threads in the row first (warp shuffle)
        // float final_l_i = warpReduceSum(l_i[i]); // Conceptual reduction
        // RO_accum[i] /= final_l_i;
        // Simplified: Assume l_i[i] holds the correct sum for the elements this thread calculated
        if (l_i[i] > 0.0f) { // Avoid division by zero
             RO_accum[i] /= l_i[i];
        } else {
             RO_accum[i] = 0.0f; // Handle case where sum is zero or negative (due to masking?)
        }
    }

    // Store Output tile
    // Map RO_accum registers back to global memory
    // Each thread writes its portion of the output tile
    // ... Store logic mapping RO_accum to O_ptr ...
     uint32_t o_start_row_global = q_start_row_block + warp_idx_q * WARP_Q;
     #pragma unroll
     for (int i = 0; i < NUM_PV_ACCUM; ++i) {
         // Determine the global row and column this accumulator corresponds to
         // This mapping is complex and depends on thread <-> MMA output element mapping
         uint32_t out_row_global; // = o_start_row_global + ...
         uint32_t out_col;        // = ...

         if (out_row_global < qo_len) {
             uint32_t o_offset = batch_id * stride_bz_o + head_id * stride_h_o + out_row_global * stride_seq_o + out_col;
             // Convert FP32 accumulator to DTypeOut (FP16)
             O[o_offset] = __float2half_rn(RO_accum[i]); // Assuming DTypeOut is half
         }
     }


    // Store LSE if needed
    if constexpr (return_lse) {
        // Map m_i and l_i registers back to global LSE tensor
        // Aggregate m_i and l_i across threads first
        // ... Store LSE logic ...
         #pragma unroll
         for (int i = 0; i < NUM_PV_ACCUM; ++i) {
            // float final_m_i = warpReduceMax(m_i[i]); // Conceptual
            // float final_l_i = warpReduceSum(l_i[i]); // Conceptual

            // Calculate LSE for the row fragment this thread is responsible for
            float lse_val = (l_i[i] > 0.f) ? (math::ptx_log2(l_i[i]) + m_i[i]) / math::log2e : -INFINITY; // Convert log2 to ln

            // Determine global row index corresponding to m_i[i]/l_i[i]
            uint32_t lse_row_global; // = o_start_row_global + ...

            // Store LSE value (potentially needs atomic add or reduction if multiple threads write to same LSE row)
            // Simplified: Assume one thread per row for LSE store after reduction
             if (lse_row_global < qo_len /* && thread_is_row_master */ ) {
                 uint32_t lse_offset = batch_id * (qo_len * num_qo_heads) + head_id * qo_len + lse_row_global;
                 Lse[lse_offset] = lse_val;
             }
         }
    }
}


// C++ function calling the kernel
torch::Tensor qk_int8_sv_f16_accum_f32_attn_sm75(
                    torch::Tensor query,
                    torch::Tensor key,
                    torch::Tensor value,
                    torch::Tensor output,
                    torch::Tensor query_scale,
                    torch::Tensor key_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse)
{
    // --- Input Checks ---
    CHECK_CUDA(query); CHECK_CUDA(key); CHECK_CUDA(value); CHECK_CUDA(output); CHECK_CUDA(query_scale); CHECK_CUDA(key_scale);
    CHECK_CONTIGUOUS(query); CHECK_CONTIGUOUS(key);
    CHECK_LASTDIM_CONTIGUOUS(value); CHECK_LASTDIM_CONTIGUOUS(output);
    CHECK_CONTIGUOUS(query_scale); CHECK_CONTIGUOUS(key_scale);

    CHECK_DTYPE(query, torch::kInt8);
    CHECK_DTYPE(key, torch::kInt8);
    CHECK_DTYPE(value, torch::kHalf); // SM75 only supports FP16 PV
    CHECK_DTYPE(query_scale, torch::kFloat32);
    CHECK_DTYPE(key_scale, torch::kFloat32);
    TORCH_CHECK(output.scalar_type() == torch::kHalf, "SM75 kernel currently only supports FP16 output.");


    CHECK_DIMS(query, 4); CHECK_DIMS(key, 4); CHECK_DIMS(value, 4); CHECK_DIMS(output, 4);
    CHECK_DIMS(query_scale, 3); CHECK_DIMS(key_scale, 3);

    const int head_dim = query.size(3);
    const int batch_size = query.size(0);

    int stride_bz_q = query.stride(0);
    int stride_bz_k = key.stride(0);
    int stride_bz_v = value.stride(0);
    int stride_bz_o = output.stride(0);

    int qo_len, kv_len, num_qo_heads, num_kv_heads;
    int stride_seq_q, stride_h_q, stride_seq_k, stride_h_k;
    int stride_seq_v, stride_h_v;
    int stride_seq_o, stride_h_o;

    if (tensor_layout == 0) // NHD
    {
        qo_len = query.size(1); kv_len = key.size(1);
        num_qo_heads = query.size(2); num_kv_heads = key.size(2);
        stride_seq_q = query.stride(1); stride_h_q = query.stride(2);
        stride_seq_k = key.stride(1); stride_h_k = key.stride(2);
        stride_seq_v = value.stride(1); stride_h_v = value.stride(2);
        stride_seq_o = output.stride(1); stride_h_o = output.stride(2);
        CHECK_SHAPE(key, batch_size, kv_len, num_kv_heads, head_dim);
        CHECK_SHAPE(value, batch_size, kv_len, num_kv_heads, head_dim);
        CHECK_SHAPE(output, batch_size, qo_len, num_qo_heads, head_dim);
    }
    else // HND
    {
        qo_len = query.size(2); kv_len = key.size(2);
        num_qo_heads = query.size(1); num_kv_heads = key.size(1);
        stride_seq_q = query.stride(2); stride_h_q = query.stride(1);
        stride_seq_k = key.stride(2); stride_h_k = key.stride(1);
        stride_seq_v = value.stride(2); stride_h_v = value.stride(1);
        stride_seq_o = output.stride(2); stride_h_o = output.stride(1);
        CHECK_SHAPE(key, batch_size, num_kv_heads, kv_len, head_dim);
        CHECK_SHAPE(value, batch_size, num_kv_heads, kv_len, head_dim);
        CHECK_SHAPE(output, batch_size, num_qo_heads, qo_len, head_dim);
    }

    if (num_qo_heads % num_kv_heads != 0) {
      std::ostringstream err_msg;
      err_msg << "num_qo_heads (" << num_qo_heads << ") must be divisible by num_kv_heads (" << num_kv_heads << ")";
      throw std::invalid_argument(err_msg.str());
    }
    const int num_kv_groups = num_qo_heads / num_kv_heads;


    torch::Tensor lse = torch::empty({0});
    if (return_lse)
    {
      lse = torch::empty({batch_size, num_qo_heads, qo_len}, query.options().dtype(torch::kFloat32));
    }

    auto output_dtype = output.scalar_type(); // Already checked it's Half

    // --- Dispatch based on template params ---
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
        DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, { // Keep gran dispatch for consistency
          DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, {
            // Output type is fixed to half for SM75 kernel path
            using DTypeOut = half;

              // SM75 specific constants (adjust based on kernel tuning)
              constexpr int CTA_Q_SM75 = 64; // Example, smaller CTA might be better for SM75
              constexpr int CTA_K_SM75 = 64;
              constexpr int WARP_Q_SM75 = 16; // Example
              constexpr int WARP_K_SM75 = 32; // Example

              constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;

              // Quant Granularity Checks (Adapt expected shapes for SM75 CTA/Warp sizes)
               if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp))
               {
                 CHECK_SHAPE(query_scale, batch_size, num_qo_heads, static_cast<long>(div_ceil(qo_len, CTA_Q_SM75) * (CTA_Q_SM75 / WARP_Q_SM75)));
                 CHECK_SHAPE(key_scale, batch_size, num_kv_heads, static_cast<long>(div_ceil(kv_len, CTA_K_SM75) * (CTA_K_SM75 / WARP_K_SM75)));
               }
               else if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread))
               {
                 // Per-thread might be very inefficient on SM75 without ldmatrix
                 // Shape check based on conceptual mapping
                 CHECK_SHAPE(query_scale, batch_size, num_qo_heads, static_cast<long>(div_ceil(qo_len, CTA_Q_SM75) * (CTA_Q_SM75 / WARP_Q_SM75) * 8)); // Assuming 8 threads per warp-q-tile row handle scales
                 CHECK_SHAPE(key_scale, batch_size, num_kv_heads, static_cast<long>(div_ceil(kv_len, CTA_K_SM75) * (CTA_K_SM75 / WARP_K_SM75) * 4)); // Assuming 4 threads per warp-k-tile row handle scales
               }

              // Calculate shared memory size needed by the SM75 kernel
              constexpr uint32_t SHMEM_PADDING_BYTES_WRAP = 16; // Match kernel padding
              constexpr uint32_t HEAD_DIM_PADDED_INT8_WRAP = HEAD_DIM + (SHMEM_PADDING_BYTES_WRAP / sizeof(int8_t));
              constexpr uint32_t HEAD_DIM_PADDED_FP16_WRAP = HEAD_DIM + (SHMEM_PADDING_BYTES_WRAP / sizeof(half));

              size_t smem_size = CTA_Q_SM75 * HEAD_DIM_PADDED_INT8_WRAP * sizeof(int8_t) +
                                 CTA_K_SM75 * HEAD_DIM_PADDED_INT8_WRAP * sizeof(int8_t) +
                                 CTA_K_SM75 * HEAD_DIM_PADDED_FP16_WRAP * sizeof(half);
             // size_t smem_o_size = (size_t)CTA_Q_SM75 * HEAD_DIM * sizeof(half); // If using shared mem for output
             // smem_size = std::max(smem_size, smem_o_size);

              auto kernel_func = qk_int8_sv_f16_accum_f32_attn_kernel_sm75<
                                      CTA_Q_SM75, CTA_K_SM75, WARP_Q_SM75, WARP_K_SM75, HEAD_DIM,
                                      static_cast<QuantGranularity>(QK_QUANT_GRAN), static_cast<QuantGranularity>(QK_QUANT_GRAN),
                                      DTypeOut, mask_mode, RETURN_LSE>;

              cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

              dim3 grid(div_ceil(qo_len, CTA_Q_SM75), num_qo_heads, batch_size);
              // Example block dim - needs tuning for SM75
              int num_warps_in_block = (CTA_Q_SM75 / WARP_Q_SM75) * (CTA_K_SM75 / WARP_K_SM75);
              dim3 block(32 * (num_warps_in_block > 0 ? num_warps_in_block : 1)); // Flatten warp layout: blockDim.x = warps * 32

              // --- Launch Kernel ---
              kernel_func<<<grid, block, smem_size>>>(
                  query.data_ptr<int8_t>(), key.data_ptr<int8_t>(), reinterpret_cast<half*>(value.data_ptr()),
                  reinterpret_cast<DTypeOut*>(output.data_ptr()), (RETURN_LSE) ? lse.data_ptr<float>() : nullptr,
                  query_scale.data_ptr<float>(), key_scale.data_ptr<float>(),
                  qo_len, kv_len, num_kv_groups,
                  stride_bz_q, stride_seq_q, stride_h_q,
                  stride_bz_k, stride_seq_k, stride_h_k,
                  stride_bz_v, stride_seq_v, stride_h_v,
                  stride_bz_o, stride_seq_o, stride_h_o,
                  sm_scale
              );
              C10_CUDA_KERNEL_LAUNCH_CHECK(); // Check for launch errors

          }); // DISPATCH_RETURN_LSE
        }); // DISPATCH_QK_QUANT_GRAN
      }); // DISPATCH_CAUSAL
    }); // DISPATCH_HEAD_DIM

    return lse;
}

#endif  // __CUDACC__

// Declaration for non-CUDA compilers (e.g. pybind11)
torch::Tensor qk_int8_sv_f16_accum_f32_attn_sm75(
                    torch::Tensor query,
                    torch::Tensor key,
                    torch::Tensor value,
                    torch::Tensor output,
                    torch::Tensor query_scale,
                    torch::Tensor key_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse);
