/*
 * Copyright (c) 2024 by SageAttention team.
 * (Adapted for SM75)
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

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "attn_cuda_sm75.h" // Include the header for SM75

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("qk_int8_sv_f16_accum_f32_attn_sm75", &qk_int8_sv_f16_accum_f32_attn_sm75, "SM75: QK int8 sv f16 accum f32 attn (direct output)");
  m.def("qk_int8_sv_f16_accum_f32_attn_sm75_smem_o", &qk_int8_sv_f16_accum_f32_attn_sm75_smem_o, "SM75: QK int8 sv f16 accum f32 attn (smem_O staging output)");
}
