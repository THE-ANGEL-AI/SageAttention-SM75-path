# ComfyUI Custom Node: SageAttention SM75 (T4)

Ускорение attention в 2× для NVIDIA T4 через INT8 тензорные ядра.

## Установка

Скопируйте папку `comfyui_nodes/` в `ComfyUI/custom_nodes/sageattention_sm75/`:

```bash
# Из корня проекта
cp -r comfyui_nodes/ /path/to/ComfyUI/custom_nodes/sageattention_sm75/

# Или через symlink (удобно для разработки):
# Windows (cmd admin):
mklink /D C:\ComfyUI\custom_nodes\sageattention_sm75 E:\Kaggle_Cloud\SageAttention-SM75-path\comfyui_nodes

# Linux/macOS:
ln -s /path/to/SageAttention-SM75-path/comfyui_nodes /path/to/ComfyUI/custom_nodes/sageattention_sm75
```

**Требования:** SageAttention-SM75 должен быть установлен (`pip install -e .` из корня проекта).

## Использование

После перезапуска ComfyUI в меню появится категория **SageAttention**:

### 🧠 SageAttention Apply (SM75 T4 INT8)

Вставьте эту ноду **между загрузчиком модели и сэмплером**:

```
[Load LTX Model] → [🧠 SageAttention Apply] → [LTX Sampler] → ...
```

**Параметры:**
| Параметр | По умолчанию | Что делает |
|---|---|---|
| `model` | (вход) | Модель к которой применяется патч |
| `smooth_k` | True | Вычитает среднее K перед attention (точнее) |
| `enable` | True | False = passthrough (без патча) |

### 🧠 SageAttention Remove

Убирает патч если нужно дальше использовать оригинальное внимание.

## Как это работает

Нода использует `model.add_object_patch()` — безопасный способ ComfyUI:
- Патч действует **только на эту модель** (не глобально)
- Не ломает другие ноды в workflow
- Авто-фолбек: custom маски / dropout / не-FP16 → оригинальное внимание

## Проверка что работает

В консоли ComfyUI должно появиться:
```
[SageAttention] ✓ Applied (smooth_k=True)
```

А GPU-утилизация (nvidia-smi) должна показать активность тензорных ядер.
