# 🧠 SageAttention-T4 — ComfyUI Custom Node

**Автор:** THEANGELAI  
**Репозиторий:** https://github.com/THE-ANGEL-AI/SageAttention-SM75-path  

Ускорение attention в ~2× для NVIDIA T4 (Turing SM75) через INT8 тензорные ядра.

---

## ⚡ Авто-установка

Клонируйте репо **прямо в custom_nodes** — нода появится автоматически после перезапуска ComfyUI:

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/THE-ANGEL-AI/SageAttention-SM75-path.git SageAttention-T4
cd SageAttention-T4
pip install -e .
```

Перезапустите ComfyUI. В меню нод появится категория **🧠 SageAttention-T4**.

> **Важно:** `pip install -e .` должен быть выполнен из папки `SageAttention-T4/` (внутри custom_nodes). Он соберёт CUDA-кернелы под вашу T4.

---

## 📦 Ноды

### 🧠 SageAttention-T4 Apply (INT8 Turbo)

Вставьте **между загрузчиком модели и сэмплером**:

```
[Load Model] → [🧠 SageAttention-T4 Apply] → [Sampler] → [VAE] → ...
```

**Параметры:**

| Параметр | По умолчанию | Что делает |
|---|---|---|
| `model` | (вход) | Модель для патча |
| `smooth_k` | True | Вычитает среднее K перед attention (точнее) |
| `enable` | True | False = passthrough (без патча, модель как есть) |

### 🧠 SageAttention-T4 Remove

Убирает патч, восстанавливая оригинальное `scaled_dot_product_attention`.

---

## 🛡️ Безопасность

- **Не глобальный monkey-patch** — используется `model.add_object_patch()`, действует только на модель прошедшую через ноду
- **Авто-фолбек** — если attention вызван с маской, dropout > 0, или не-FP16 типом → прозрачно передаётся оригинальному `sdpa`
- **Не ломает другие ноды** — другие части workflow не затронуты

---

## ✅ Проверка что работает

В консоли ComfyUI (запущенной с `--verbose`) должно появиться:

```
[SageAttention] SageAttention imported successfully
[SageAttention] SageAttention applied (smooth_k=True)
```

В `nvidia-smi` GPU должна показывать высокую утилизацию compute (не копирование).
