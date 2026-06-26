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
#define MMA_QK_K_SM75 16 // INT8 MMA K dim (m8n8k16 for SM75 Turing)

#define MMA_SV_M_SM75 8  // m8n8k4 fp16 MMA M dim (SM75: m8, not m16)
#define MMA_SV_N_SM75 8
#define MMA_SV_K_SM75 4 // m8n8k4 fp16 MMA K dim (only valid fp16 shape on SM75)

// --- SM75 Kernel Implementation ---
template< uint32_t CTA_Q, uint32_t CTA_K, uint32_t WARP_Q, uint32_t WARP_K, uint32_t head_dim,
          QuantGranularity Q_GRAN, QuantGranularity K_GRAN,
          typename DTypeOut, MaskMode mask_mode, bool return_lse,
          bool USE_SMEM_O_OUTPUT = false>
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

    // --- Compile-time shape validation ---
    // head_dim must be divisible by MMA tile sizes (INT8 K=16, FP16 N=8)
    static_assert(head_dim % MMA_QK_K_SM75 == 0,
        "head_dim must be divisible by MMA_QK_K_SM75 (16) for INT8 MMA K dimension");
    static_assert(head_dim % MMA_SV_N_SM75 == 0,
        "head_dim must be divisible by MMA_SV_N_SM75 (8) for FP16 PV MMA N dimension");
    // CTA tile must subdivide evenly into warp tiles
    static_assert(CTA_Q % WARP_Q == 0,
        "CTA_Q must be divisible by WARP_Q");
    static_assert(CTA_K % WARP_K == 0,
        "CTA_K must be divisible by WARP_K");

    // --- Thread/Block Indexing ---
    const uint32_t lane_id = threadIdx.x % 32; // Lane index within the warp (0-31)
    const uint32_t warp_id_in_block = threadIdx.x / 32; // Warp index within the CTA
    const uint32_t num_warps_per_block = blockDim.x / 32;
    (void)num_warps_per_block; // Reserved for future assert/check

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
    // Per-warp P buffer for PV MMA: each warp stores its 8×8 fp16 softmax output
    // before loading via ldmatrix for correct MMA thread→data distribution.
    constexpr int P_TILE_ELEMS = MMA_QK_M_SM75 * MMA_QK_N_SM75; // 8×8 = 64 halfs
    half*   smem_P = reinterpret_cast<half*>(smem_V + CTA_K * HEAD_DIM_PADDED_FP16);
    // Output staging buffer (only when USE_SMEM_O_OUTPUT=true).
    // Layout: CTA_Q rows × head_dim columns, row-major, half precision, no padding.
    half*   smem_O = nullptr;
    if constexpr (USE_SMEM_O_OUTPUT) {
        smem_O = reinterpret_cast<half*>(smem_P + (CTA_Q / WARP_Q) * (CTA_K / WARP_K) * P_TILE_ELEMS);
    }
    // --- Register Allocation ---
    // Tile counts (kernel-scope, used throughout)
    constexpr int NUM_M_TILES = WARP_Q / MMA_QK_M_SM75;   // 2
    constexpr int NUM_N_TILES = WARP_K / MMA_QK_N_SM75;   // 2
    constexpr int NUM_N_V_TILES = head_dim / MMA_SV_N_SM75; // V column sub-tiles (e.g. 8 for hd=64)

    // QK Accumulator (INT8 MMA -> INT32)
    // m8n8k16: each thread holds 2 int32 accumulators (PTX: {%0, %1})
    constexpr int NUM_QK_ACCUM = 2;
    int32_t RS_accum[NUM_QK_ACCUM];

    // PV Accumulator (FP16 MMA -> FP32)
    // SM75: m8n8k4 => 2 FP32 outputs per MMA call.
    // NUM_N_V_TILES = head_dim / MMA_SV_N V column sub-tiles.
    // Each (mq, fk) sub-tile needs 2 accumulators.
    // 2D layout: RO_accum[mq][fk*2+0..1] stored flat.
    constexpr int NUM_PV_ACCUM_PER_M_TILE = NUM_N_V_TILES * 2;
    constexpr int NUM_PV_ACCUM_TOTAL = NUM_M_TILES * NUM_PV_ACCUM_PER_M_TILE;
    float RO_accum[NUM_PV_ACCUM_TOTAL];

    // --- Online Softmax State ---
    // SM75 m8n8k4 MMA: each thread handles 1 row per mq sub-tile (2 values c0,c1).
    // m_i[mq] and l_i[mq] track the running max/denominator for the row.
    float m_i[NUM_M_TILES]; // Running max per mq sub-tile
    float l_i[NUM_M_TILES]; // Running sum per mq sub-tile

    // --- Initialization (once, before all K tiles) ---
    #pragma unroll
    for (int i = 0; i < NUM_PV_ACCUM_TOTAL; ++i) {
        RO_accum[i] = 0.0f;
    }
    #pragma unroll
    for (int i = 0; i < NUM_M_TILES; ++i) {
        m_i[i] = -1e30f;
        l_i[i] = 0.0f;
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
        (void)current_k_block_len; // Reserved for boundary-aware load/loop logic

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

        // --- QK^T Computation (using INT8 MMA) with Online Softmax ---
        // Each warp computes a WARP_Q x WARP_K tile of S = QK^T
        // SM75 m8n8k16 MMA: all 32 threads participate, organized into 4 groups of 8.
        // Each thread i computes row r = i % 8, with 2 values at cols c0 = (i/8)*2, c1 = (i/8)*2+1.
        // Softmax warp reductions: threads {r, r+8, r+16, r+24} share row r.
        uint32_t q_start_warp = warp_idx_q * WARP_Q;
        uint32_t k_start_warp = warp_idx_k * WARP_K;

        #pragma unroll
        for(int mq = 0; mq < NUM_M_TILES; ++mq) {
            // NOTE: RO_accum[mq][*], m_i[mq], l_i[mq] CARRY OVER across K tiles
            // (online softmax). Only initialized once before the K-tile loop.

            #pragma unroll
            for(int nk = 0; nk < NUM_N_TILES; ++nk) {
                // --- QK MMA: compute S[mq, nk] = Q[mq] × K[nk]^T ---
                #pragma unroll
                for(int acc_idx = 0; acc_idx < NUM_QK_ACCUM; ++acc_idx) RS_accum[acc_idx] = 0;

                #pragma unroll
                for(int hk = 0; hk < head_dim / MMA_QK_K_SM75; ++hk) {
                    // Load Q fragment (int8, row-major, A operand for MMA)
                    int8_t* smem_Q_ptr = smem_Q + (q_start_warp + mq * MMA_QK_M_SM75 + lane_id % 8) * HEAD_DIM_PADDED_INT8 + hk * MMA_QK_K_SM75;
                    uint32_t q_frag_reg_load[4] = {0, 0, 0, 0};
                    mma::ldmatrix_m8n8x4(q_frag_reg_load, smem_Q_ptr);
                    uint32_t q_frag_reg[1] = {q_frag_reg_load[0]};

                    // Load K fragment (int8, row-major in smem = col-major for B operand)
                    int8_t* smem_K_ptr = smem_K + (k_start_warp + nk * MMA_QK_N_SM75 + lane_id % 8) * HEAD_DIM_PADDED_INT8 + hk * MMA_QK_K_SM75;
                    uint32_t k_frag_reg_load[4] = {0, 0, 0, 0};
                    mma::ldmatrix_m8n8x4(k_frag_reg_load, smem_K_ptr);
                    uint32_t k_frag_reg[1] = {k_frag_reg_load[0]};

                    // MMA m8n8k16 int8→int32
                    mma::mma_sync_m8n8k16_row_col_s8s8s32<mma::MMAMode::kInplaceUpdate>(RS_accum, q_frag_reg, k_frag_reg);
                } // End hk loop (head_dim K dimension)

                // --- Online Softmax (FlashAttention-style) ---
                // Thread i: row r = lane_id % 8, cols c0 = nk*8 + (lane_id/8)*2, c1 = c0 + 1
                uint32_t global_q_idx = q_start_row_block + q_start_warp + mq * MMA_QK_M_SM75 + (lane_id % 8);
                uint32_t s_tile_start_col = k_start_row_block + k_start_warp + nk * MMA_QK_N_SM75;
                uint32_t global_k_idx_0 = s_tile_start_col + (lane_id / 8) * 2;
                uint32_t global_k_idx_1 = global_k_idx_0 + 1;

                // Convert int32 → float, apply dequant scale
                float s0 = __int2float_rn(RS_accum[0]) * current_dequant_scale;
                float s1 = __int2float_rn(RS_accum[1]) * current_dequant_scale;

                // Apply masking (Q boundary, K boundary, causal)
                bool q_oob = (global_q_idx >= qo_len);
                if (q_oob) { s0 = -1e30f; s1 = -1e30f; }
                if (is_boundary_k_iter) {
                    if (global_k_idx_0 >= kv_len) s0 = -1e30f;
                    if (global_k_idx_1 >= kv_len) s1 = -1e30f;
                }
                if constexpr (mask_mode == MaskMode::kCausal) {
                    if (global_k_idx_0 > global_q_idx) s0 = -1e30f;
                    if (global_k_idx_1 > global_q_idx) s1 = -1e30f;
                }

                // 1. Local max of 2 values for this thread's row (dequantized only)
                float m_local = fmaxf(s0, s1);

                // 2. Scale max by sm_scale (includes log2e) before warp reduce
                //    Per SM80 pattern: max_i(s_i) * sm_scale = max_i(s_i * sm_scale)
                m_local *= sm_scale;

                // 3. Warp-reduce max across 4 threads sharing the same row
                //    Threads {r, r+8, r+16, r+24} share row r = lane_id % 8
                m_local = fmaxf(m_local, __shfl_xor_sync(0xffffffff, m_local, 8));
                m_local = fmaxf(m_local, __shfl_xor_sync(0xffffffff, m_local, 16));

                // 4. Update online softmax state for this row (mq sub-tile)
                float m_prev = m_i[mq];
                float m_new = fmaxf(m_prev, m_local);
                m_i[mq] = m_new;

                // 5. Renormalize: o_scale = exp2(m_prev - m_new)
                float o_scale = math::ptx_exp2(m_prev - m_new);
                l_i[mq] *= o_scale;

                // Renormalize PV accumulators for this mq sub-tile
                #pragma unroll
                for(int fk = 0; fk < NUM_N_V_TILES; ++fk) {
                    RO_accum[mq * NUM_PV_ACCUM_PER_M_TILE + fk * 2 + 0] *= o_scale;
                    RO_accum[mq * NUM_PV_ACCUM_PER_M_TILE + fk * 2 + 1] *= o_scale;
                }

                // 6. Compute P = exp2(S * sm_scale - m_new) (attention weights)
                //    sm_scale already includes log2e; fmaf: s0*sm_scale + (-m_new)
                float p0 = math::ptx_exp2(fmaf(s0, sm_scale, -m_new));
                float p1 = math::ptx_exp2(fmaf(s1, sm_scale, -m_new));
                if (s0 <= -1e29f) p0 = 0.0f;
                if (s1 <= -1e29f) p1 = 0.0f;

                // 7. Accumulate denominator l_i across the row
                float l_local = p0 + p1;
                l_local += __shfl_xor_sync(0xffffffff, l_local, 8);
                l_local += __shfl_xor_sync(0xffffffff, l_local, 16);
                l_i[mq] += l_local;

                // 8. Store P to shared memory for PV MMA (m8n8k4, A=1 register/thread).
                //    Thread i writes to row r = lane_id % 8, cols c0 = (i/8)*2, c1 = c0+1.
                half* smem_P_warp = smem_P + warp_id_in_block * P_TILE_ELEMS;
                uint32_t r = lane_id % 8;
                smem_P_warp[r * MMA_QK_N_SM75 + (lane_id / 8) * 2 + 0] = __float2half_rn(p0);
                smem_P_warp[r * MMA_QK_N_SM75 + (lane_id / 8) * 2 + 1] = __float2half_rn(p1);
                __syncwarp(); // Ensure all 32 threads finished writing before ldmatrix

                // --- PV Computation (FP16 MMA m8n8k4): RO += P × V ---
                // P is 8×8, MMA K=4 → split K dimension into 2 passes (pv_k = 0, 1)
                constexpr int PV_K_STEPS = MMA_QK_N_SM75 / MMA_SV_K_SM75; // 8 / 4 = 2
                #pragma unroll
                for (int pv_k = 0; pv_k < PV_K_STEPS; ++pv_k) {

                    // Load K=4 chunk of P (2 fp16 values per thread = 1 b32 register)
                    half* smem_P_row_ptr = smem_P_warp + r * MMA_QK_N_SM75 + pv_k * MMA_SV_K_SM75;
                    uint32_t p_frag_reg_k[1] = {0};
                    mma::ldmatrix_m8n8x1(p_frag_reg_k, smem_P_row_ptr);

                    #pragma unroll
                    for(int fk = 0; fk < NUM_N_V_TILES; ++fk) {
                        // Load K=4 chunk of V (row-shifted by pv_k * MMA_SV_K_SM75)
                        half* smem_V_ptr = smem_V + (k_start_warp + nk * MMA_QK_N_SM75 + pv_k * MMA_SV_K_SM75 + lane_id % 8) * HEAD_DIM_PADDED_FP16 + fk * MMA_SV_N_SM75;
                        uint32_t v_frag_reg_k[1] = {0};
                        mma::ldmatrix_m8n8x1_trans(v_frag_reg_k, smem_V_ptr);

                        // PV MMA: RO_accum[mq][fk*2..fk*2+1] += P_chunk × V_chunk
                        mma::mma_sync_m8n8k4_row_col_f16f16f32<mma::MMAMode::kInplaceUpdate>(
                            RO_accum + mq * NUM_PV_ACCUM_PER_M_TILE + fk * 2,
                            p_frag_reg_k,
                            v_frag_reg_k);
                    } // End fk loop (V column sub-tiles)
                } // End pv_k loop (K dimension split)
            } // End nk loop (S tile N dimension)
        } // End mq loop (S tile M dimension)
        __syncthreads(); // Sync after finishing work with current K/V tile before loading next
    } // End K tile loop

    // --- Final Normalization ---
    // Normalize: O[mq] = RO_accum[mq][*] / l_i[mq]. l_i already warp-reduced.
    #pragma unroll
    for(int mq = 0; mq < NUM_M_TILES; ++mq) {
        float l_rcp = (l_i[mq] > 0.0f) ? math::ptx_rcp(l_i[mq]) : 0.0f;
        #pragma unroll
        for(int fk = 0; fk < NUM_N_V_TILES; ++fk) {
            RO_accum[mq * NUM_PV_ACCUM_PER_M_TILE + fk * 2 + 0] *= l_rcp;
            RO_accum[mq * NUM_PV_ACCUM_PER_M_TILE + fk * 2 + 1] *= l_rcp;
        }
    }

    // --- Output: Direct Write or smem_O Staging ---
    uint32_t o_start_row_warp = warp_idx_q * WARP_Q;
    if constexpr (USE_SMEM_O_OUTPUT) {
        // Path A: Stage to smem_O → __syncthreads() → coalesced write to global
        #pragma unroll
        for(int mq = 0; mq < NUM_M_TILES; ++mq) {
            uint32_t local_row = mq * MMA_SV_M_SM75 + (lane_id % 8);
            uint32_t smem_row = o_start_row_warp + local_row;
            #pragma unroll
            for(int fk = 0; fk < NUM_N_V_TILES; ++fk) {
                uint32_t col0 = fk * MMA_SV_N_SM75 + (lane_id / 8) * 2;
                uint32_t col1 = col0 + 1;
                smem_O[smem_row * head_dim + col0] = __float2half_rn(RO_accum[mq * NUM_PV_ACCUM_PER_M_TILE + fk * 2 + 0]);
                smem_O[smem_row * head_dim + col1] = __float2half_rn(RO_accum[mq * NUM_PV_ACCUM_PER_M_TILE + fk * 2 + 1]);
            }
        }
        __syncthreads(); // All warps finished writing smem_O

        // Coalesced write: each thread reads a contiguous chunk from smem_O
        for (int i = threadIdx.x; i < CTA_Q * head_dim; i += blockDim.x) {
            uint32_t local_row = i / head_dim;
            uint32_t local_col = i % head_dim;
            uint32_t global_row = q_start_row_block + local_row;
            if (global_row < qo_len) {
                uint32_t o_offset = batch_id * stride_bz_o + head_id * stride_h_o + global_row * stride_seq_o + local_col;
                O[o_offset] = smem_O[local_row * head_dim + local_col];
            }
        }
    } else {
        // Path B: Direct scattered write from threads to global O
        #pragma unroll
        for(int mq = 0; mq < NUM_M_TILES; ++mq) {
            uint32_t global_row = q_start_row_block + o_start_row_warp + mq * MMA_SV_M_SM75 + (lane_id % 8);
            if (global_row < qo_len) {
                #pragma unroll
                for(int fk = 0; fk < NUM_N_V_TILES; ++fk) {
                    uint32_t col0 = fk * MMA_SV_N_SM75 + (lane_id / 8) * 2;
                    uint32_t col1 = col0 + 1;
                    uint32_t o_offset = batch_id * stride_bz_o + head_id * stride_h_o + global_row * stride_seq_o;
                    O[o_offset + col0] = __float2half_rn(RO_accum[mq * NUM_PV_ACCUM_PER_M_TILE + fk * 2 + 0]);
                    O[o_offset + col1] = __float2half_rn(RO_accum[mq * NUM_PV_ACCUM_PER_M_TILE + fk * 2 + 1]);
                }
            }
        }
    }

    // --- Store LSE if needed ---
    // Each row's LSE is computed from m_i[mq] and l_i[mq]. 4 threads share the same
    // row — only threads with lane_id < 8 write (one value per unique row).
    if constexpr (return_lse) {
         #pragma unroll
         for (int mq = 0; mq < NUM_M_TILES; ++mq) {
            if (lane_id < 8) {
                float lse_val = (l_i[mq] > 0.f) ? (math::ptx_log2(l_i[mq]) + m_i[mq]) / math::log2e : -1e30f;
                uint32_t lse_row_global = q_start_row_block + o_start_row_warp + mq * MMA_SV_M_SM75 + lane_id;
                if (lse_row_global < qo_len) {
                    uint32_t lse_offset = batch_id * (qo_len * num_qo_heads) + head_id * qo_len + lse_row_global;
                    Lse[lse_offset] = lse_val;
                }
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
              constexpr int WARP_K_SM75 = 16; // WARP_K=16 gives 2 N-sub-tiles (16/MMA_SV_N=2), fits NUM_PV_ACCUM=4

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

              constexpr int P_TILE_ELEMS_WRAP = 8 * 8; // MMA_QK_M × MMA_QK_N = 64 halfs per warp
              constexpr int NUM_WARPS_WRAP = (CTA_Q_SM75 / WARP_Q_SM75) * (CTA_K_SM75 / WARP_K_SM75);
              size_t smem_size = CTA_Q_SM75 * HEAD_DIM_PADDED_INT8_WRAP * sizeof(int8_t) +
                                 CTA_K_SM75 * HEAD_DIM_PADDED_INT8_WRAP * sizeof(int8_t) +
                                 CTA_K_SM75 * HEAD_DIM_PADDED_FP16_WRAP * sizeof(half) +
                                 NUM_WARPS_WRAP * P_TILE_ELEMS_WRAP * sizeof(half);  // per-warp P buffer

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

// --- SM75 with smem_O staging variant ---
torch::Tensor qk_int8_sv_f16_accum_f32_attn_sm75_smem_o(
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
    // --- Input Checks (identical to base variant) ---
    CHECK_CUDA(query); CHECK_CUDA(key); CHECK_CUDA(value); CHECK_CUDA(output); CHECK_CUDA(query_scale); CHECK_CUDA(key_scale);
    CHECK_CONTIGUOUS(query); CHECK_CONTIGUOUS(key);
    CHECK_LASTDIM_CONTIGUOUS(value); CHECK_LASTDIM_CONTIGUOUS(output);
    CHECK_CONTIGUOUS(query_scale); CHECK_CONTIGUOUS(key_scale);
    CHECK_DTYPE(query, torch::kInt8);
    CHECK_DTYPE(key, torch::kInt8);
    CHECK_DTYPE(value, torch::kHalf);
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
    if (return_lse) {
      lse = torch::empty({batch_size, num_qo_heads, qo_len}, query.options().dtype(torch::kFloat32));
    }

    // --- Dispatch with USE_SMEM_O_OUTPUT = true ---
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
        DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
          DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, {
            using DTypeOut = half;
              constexpr int CTA_Q_SM75 = 64;
              constexpr int CTA_K_SM75 = 64;
              constexpr int WARP_Q_SM75 = 16;
              constexpr int WARP_K_SM75 = 16;
              constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;

              // Shape checks (same as base)
              if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp)) {
                 CHECK_SHAPE(query_scale, batch_size, num_qo_heads, static_cast<long>(div_ceil(qo_len, CTA_Q_SM75) * (CTA_Q_SM75 / WARP_Q_SM75)));
                 CHECK_SHAPE(key_scale, batch_size, num_kv_heads, static_cast<long>(div_ceil(kv_len, CTA_K_SM75) * (CTA_K_SM75 / WARP_K_SM75)));
              } else if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread)) {
                 CHECK_SHAPE(query_scale, batch_size, num_qo_heads, static_cast<long>(div_ceil(qo_len, CTA_Q_SM75) * (CTA_Q_SM75 / WARP_Q_SM75) * 8));
                 CHECK_SHAPE(key_scale, batch_size, num_kv_heads, static_cast<long>(div_ceil(kv_len, CTA_K_SM75) * (CTA_K_SM75 / WARP_K_SM75) * 4));
              }

              constexpr uint32_t SHMEM_PADDING_BYTES_WRAP = 16;
              constexpr uint32_t HEAD_DIM_PADDED_INT8_WRAP = HEAD_DIM + (SHMEM_PADDING_BYTES_WRAP / sizeof(int8_t));
              constexpr uint32_t HEAD_DIM_PADDED_FP16_WRAP = HEAD_DIM + (SHMEM_PADDING_BYTES_WRAP / sizeof(half));
              constexpr int P_TILE_ELEMS_WRAP = 8 * 8;
              constexpr int NUM_WARPS_WRAP = (CTA_Q_SM75 / WARP_Q_SM75) * (CTA_K_SM75 / WARP_K_SM75);
              // smem_O adds CTA_Q × head_dim halfs for output staging
              size_t smem_size = CTA_Q_SM75 * HEAD_DIM_PADDED_INT8_WRAP * sizeof(int8_t) +
                                 CTA_K_SM75 * HEAD_DIM_PADDED_INT8_WRAP * sizeof(int8_t) +
                                 CTA_K_SM75 * HEAD_DIM_PADDED_FP16_WRAP * sizeof(half) +
                                 NUM_WARPS_WRAP * P_TILE_ELEMS_WRAP * sizeof(half) +
                                 CTA_Q_SM75 * HEAD_DIM * sizeof(half);  // + smem_O

              // USE_SMEM_O_OUTPUT = true
              auto kernel_func = qk_int8_sv_f16_accum_f32_attn_kernel_sm75<
                  CTA_Q_SM75, CTA_K_SM75, WARP_Q_SM75, WARP_K_SM75, HEAD_DIM,
                  static_cast<QuantGranularity>(QK_QUANT_GRAN), static_cast<QuantGranularity>(QK_QUANT_GRAN),
                  DTypeOut, mask_mode, RETURN_LSE, true>;  // USE_SMEM_O_OUTPUT=true

              cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

              dim3 grid(div_ceil(qo_len, CTA_Q_SM75), num_qo_heads, batch_size);
              int num_warps_in_block = (CTA_Q_SM75 / WARP_Q_SM75) * (CTA_K_SM75 / WARP_K_SM75);
              dim3 block(32 * (num_warps_in_block > 0 ? num_warps_in_block : 1));

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
              C10_CUDA_KERNEL_LAUNCH_CHECK();
          });
        });
      });
    });

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

torch::Tensor qk_int8_sv_f16_accum_f32_attn_sm75_smem_o(
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
