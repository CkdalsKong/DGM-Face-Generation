# DGM Spring 2026 — Face Generation Challenge

**Team:** Changmin_Lee | **Student ID:** 20255343 | **Best leaderboard FID:** 37.49

## Final Submission

The submitted 1,000 images were generated from a **StyleGAN3-R** checkpoint fine-tuned on CelebV-HQ with ADA augmentation and horizontal mirroring.

| Setting | Value |
|---------|-------|
| Base model | `stylegan3-r-ffhqu-256x256.pkl` (NVIDIA FFHQ-U pretrained) |
| Resume from | StyleGAN3 noaug checkpoint at kimg 604 |
| Additional training | 163 kimg, ADA aug, mirror=1, 2×GPU, batch=128 |
| Generation | `gen_images.py`, seeds 0–999, trunc ψ=1.0 |
| Resolution | 256×256 PNG |

### Reproduce the submission

```bash
# 1. Clone StyleGAN3
git clone https://github.com/NVlabs/stylegan3.git
cd stylegan3

# 2. Apply Blackwell/PyTorch 2.12 patches
cp ../patches/stylegan3_custom_ops.py torch_utils/custom_ops.py
cp ../patches/stylegan3_misc.py torch_utils/misc.py
cp ../patches/stylegan3_train.py train.py

# 3. Download checkpoint (link in Releases)
# Place at: checkpoints/stylegan3_aug_kimg163.pkl

# 4. Generate
python gen_images.py \
  --outdir=submission/gen \
  --trunc=1.0 \
  --seeds=0-999 \
  --network=checkpoints/stylegan3_aug_kimg163.pkl
```

---

## Environment

- **GPU:** NVIDIA RTX PRO 6000 Blackwell (sm_120, 97 GB VRAM)
- **PyTorch:** 2.12.0.dev20260407+cu128 (nightly)
- **CUDA:** 12.8
- **Python:** 3.10
- **OS:** Ubuntu 24.04

```bash
pip install -r requirements.txt
```

---

## Blackwell + PyTorch 2.12 Patches

Running StyleGAN2-ADA and StyleGAN3 on Blackwell (sm_120) with PyTorch 2.12 nightly required three patches. All patched files are in `patches/`.

### 1. `custom_ops.py` — CUDA kernel loading

PyTorch 2.12 returns the module object directly from `torch.utils.cpp_extension.load()`.
The original code calls `importlib.import_module()` afterward, which raises `ModuleNotFoundError`.

**Fix:** Replace `importlib.import_module(module_name)` with the return value of `load()`.

```python
# Before
torch.utils.cpp_extension.load(name=module_name, ...)
module = importlib.import_module(module_name)  # fails in PyTorch 2.12

# After
module = torch.utils.cpp_extension.load(name=module_name, ...)
```

### 2. `misc.py` — InfiniteSampler

`torch.utils.data.Sampler.__init__` no longer accepts a `dataset` argument in PyTorch 2.12.

```python
# Before
super().__init__(dataset)

# After
super().__init__()
```

### 3. `misc.py` — copy_params_and_buffers (StyleGAN3 only)

When resuming with a different channel config, shape mismatches cause `RuntimeError`.
Added a shape check to skip mismatched tensors (random init fallback).

```python
# Before
if name in src_tensors:
    tensor.copy_(src_tensors[name].detach())

# After
if name in src_tensors and src_tensors[name].shape == tensor.shape:
    tensor.copy_(src_tensors[name].detach())
```

### 4. `train.py` — Decouple G/D channel base (StyleGAN3 large experiment)

To use `cbase=32768` for G while keeping D at `cbase=16384`:

```python
# Before
c.G_kwargs.channel_base = c.D_kwargs.channel_base = opts.cbase

# After
c.G_kwargs.channel_base = opts.cbase
c.D_kwargs.channel_base = 16384
```

---

## Experiment Summary

| Model | Leaderboard FID↓ | KID↓ | TopPR↑ |
|-------|-----------------|------|--------|
| SD1.5 + LoRA | 65.62 | 0.042 | 0.696 |
| RV6 + LoRA (best) | 68.53 | 0.044 | 0.554 |
| StyleGAN2 (1k) | 44.52 | 0.017 | 0.905 |
| StyleGAN3 noaug | 39.30 | 0.013 | 0.872 |
| **StyleGAN3 aug+mirror ★** | **37.49** | **0.011** | **0.865** |

### Training commands

**StyleGAN3 noaug (baseline):**
```bash
CUDA_VISIBLE_DEVICES=0 python stylegan3/train.py \
  --outdir=runs/stylegan3_noaug \
  --cfg=stylegan3-r \
  --data=CelebV-HQ/celebvhq_256.zip \
  --resume=pretrained/stylegan3-r-ffhqu-256x256.pkl \
  --gpus=1 --batch=64 --gamma=2.0 --mirror=1 \
  --cbase=16384 --cmax=512 \
  --kimg=1000 --snap=40 --metrics=none --aug=noaug
```

**StyleGAN3 aug+mirror (final, resume from noaug-604):**
```bash
CUDA_VISIBLE_DEVICES=0,1 python stylegan3/train.py \
  --outdir=runs/stylegan3_aug \
  --cfg=stylegan3-r \
  --data=CelebV-HQ/celebvhq_256.zip \
  --resume=runs/stylegan3_noaug/.../network-snapshot-000604.pkl \
  --gpus=2 --batch=128 --gamma=2.0 --mirror=1 \
  --cbase=16384 --cmax=512 \
  --kimg=400 --snap=40 --metrics=none --aug=ada
```

---

## Checkpoint

The final checkpoint (`stylegan3_aug_kimg163.pkl`, 213 MB) is available in the [Releases](../../releases) tab.

- Seeds used: 0–999
- Truncation: ψ = 1.0
