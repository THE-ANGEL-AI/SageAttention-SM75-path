<p align="center">
  <a href="https://boosty.to/the_angel/donate" target="_blank">
    <img src="https://img.shields.io/badge/💖_Поддержать_проект-Донат_на_Boosty-ff6b6b?style=for-the-badge&logo=boosty" alt="Поддержать проект">
  </a>
</p>

---

<h1 align="center">⚡ SageAttention-SM75</h1>
<h3 align="center">Ускоренное внимание для NVIDIA T4 (Turing) · INT8 QK · FP16 PV · FP32 аккумуляция</h3>

<p align="center">
  <em>Форк официального <a href="https://github.com/thu-ml/SageAttention">SageAttention</a> с полной оптимизацией под SM75 (Turing) — NVIDIA T4, RTX 2080, Quadro RTX</em>
</p>

---

## ⚠️ Статус: Предварительная версия

> **Тестирование активно ведётся.** Кернел проходит юнит-тесты корректности, но end-to-end тесты на видео-моделях (LTX-Video, CogVideoX) и GGUF-инференсе ещё в процессе. Возможны изменения API и доработки производительности.

---

## 🎯 Для чего этот форк

Оригинальный SageAttention **не поддерживает SM75 (Turing)** — на T4 он просто не собирается. Мы исправили это и написали полноценный CUDA-кернел для архитектуры Turing, использующий:

- **INT8 тензорные ядра** для QK<sup>⊤</sup> (m8n8k16 — 130 TOPS)
- **FP16 тензорные ядра** для PV (m8n8k16 — 65 TFLOPS)
- **FP32 аккумуляцию** для точности
- **Полный FlashAttention-совместимый softmax** (онлайн, warp-редукции)
- **LSE (log-sum-exp)** для Ring Attention / xDiT

---

## 🚀 Что мы изменили и добавили

### 🔧 CUDA-кернел SM75 (`csrc/qattn/attn_cuda_sm75.h`)

| Компонент | Что сделано |
|---|---|
| **MMA-обёртки** (`csrc/mma.cuh`) | m8n8k16 INT8 + FP16 для SM75 с архитектурными гардами и рантайм-ассертами |
| **QK MMA** | INT8 m8n8k16 с ldmatrix-загрузкой Q/K фрагментов |
| **Онлайн-softmax** | Полный FlashAttention: running max/denominator, warp-редукции через `__shfl_xor_sync` |
| **P → smem → ldmatrix** | Softmax-выход пишется в shared memory, перезагружается через ldmatrix для корректной MMA-раскладки |
| **PV MMA** | FP16 m8n8k16 — fk-цикл итерирует **столбцы** V (head_dim/8 суб-тайлов) |
| **2D RO_accum** | Раздельные аккумуляторы для каждого mq-подтайла, онлайн-softmax через K-тайлы |
| **Output store** | Прямая запись в global O (без smem_O roundtrip) + smem_O staging вариант для бенчмарка |
| **static_assert** | Проверки на этапе компиляции: делимость head_dim, CTA_Q/WARP_Q, CTA_K/WARP_K |
| **LSE** | Log-sum-exp возврат с warp-редукциями для Ring Attention |

### 🐍 Python-уровень (`sageattention/core.py`)

| Компонент | Что сделано |
|---|---|
| **Авто-диспетчер** | `sageattn()` → SM75 → `sageattn_qk_int8_pv_fp16_cuda_sm75()` |
| **BF16→FP16** | Авто-каст для Turing (не поддерживает BF16) |
| **Padding** | Авто-подгон head_dim под 16 (INT8 MMA K) |
| **Починены баги** | try/except, stray-код, дублирующиеся блоки, экранированные docstring |

### 🧪 Тестирование

| Файл | Что проверяет |
|---|---|
| `test_sm75_kernel.py` | 4 конфига head_dim, GQA, causal, LSE, edge cases, back-to-back переиспользование тензора |
| `bench/bench_sm75_output_store.py` | Микро-бенчмарк: smem_O staging vs прямой вывод (13 конфигов) |

### 🔌 Интеграция в ComfyUI

| Файл | Назначение |
|---|---|
| `scripts/ltxv_sageattn_patch.py` | Monkey-patch `F.scaled_dot_product_attention` → SageAttention для LTX-Video |
| `scripts/build_sm75_kaggle.sh` | Скрипт сборки для Kaggle T4×2 |

---

## 📊 Ожидаемый прирост скорости

### На одной T4 (FP16 → INT8 attention)

| Длина контекста | Ускорение attention | End-to-end (видео-модели) |
|---|---|---|
| 2K | ~1.3× | +10-15% |
| 8K | ~1.8× | +30-40% |
| 16K | ~2.5× | +60-75% |

### На двух T4 с MultiGPU (Pipeline Parallelism)

При правильном распределении слоёв по GPU + INT8 внимании:
- **LTX-Video 1280×720×15сек**: с ~400 сек/итерацию → **~45-70 сек** (5-9× общий прирост — в основном за счёт устранения PCIe-spill в системную RAM)

---

## 📦 Установка

```bash
# Клонируем форк
git clone https://github.com/THE-ANGEL-AI/SageAttention-SM75-path.git
cd SageAttention-SM75-path

# Собираем SM75 расширение
python setup.py build_ext --inplace

# Или на Kaggle T4×2:
bash scripts/build_sm75_kaggle.sh

# Проверяем
python test_sm75_kernel.py
```

**Требования:** Python ≥3.9, PyTorch ≥2.3, CUDA ≥11.8, NVIDIA T4 (или другой SM75 GPU)

---

## 🔧 Быстрый старт

```python
from sageattention import sageattn
import torch.nn.functional as F

# Вариант 1: Прямой вызов
output = sageattn(q, k, v, tensor_layout="HND", is_causal=False)

# Вариант 2: Monkey-patch (все sdpa вызовы → SageAttention)
F.scaled_dot_product_attention = sageattn
```

**Интеграция в ComfyUI (LTX-Video):**
```python
# В ноде загрузчика LTX-Video добавьте:
from ltxv_sageattn_patch import apply_patch
apply_patch(smooth_k=True, qk_quant_gran='per_warp')
```

---

## 📁 Структура проекта

```
SageAttention-SM75-path/
├── csrc/qattn/
│   ├── attn_cuda_sm75.h       # ← основной SM75-кернел
│   ├── pybind_sm75.cpp         # ← pybind-регистрация
│   └── qk_int_sv_f16_cuda_sm75.cu
├── csrc/mma.cuh                # ← SM75 MMA-обёртки (int8 + fp16)
├── sageattention/core.py       # ← авто-диспетчер + SM75-функция
├── test_sm75_kernel.py         # ← юнит-тесты
├── bench/bench_sm75_output_store.py  # ← бенчмарк output store
├── scripts/
│   ├── build_sm75_kaggle.sh    # ← сборка на Kaggle
│   └── ltxv_sageattn_patch.py  # ← патч для ComfyUI LTX-Video
└── setup.py                    # ← сборка с флагом HAS_SM75
```

---

## 🙏 Благодарности

Оригинальный SageAttention: [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention) — Jintao Zhang, Jia Wei, Haofeng Huang, Pengle Zhang, Jun Zhu, Jianfei Chen.

Форк основан на [XUANNISSAN/SageAttention-SM75-path](https://github.com/XUANNISSAN/SageAttention-SM75-path).

---

<p align="center">
  <a href="https://boosty.to/the_angel/donate" target="_blank">
    <img src="https://img.shields.io/badge/💖_Поддержать_проект-Донат_на_Boosty-ff6b6b?style=for-the-badge&logo=boosty" alt="Поддержать проект">
  </a>
</p>

<p align="center">
  <sub>⚡ SageAttention-SM75 · Предварительная версия · Тестирование в процессе</sub>
</p>
