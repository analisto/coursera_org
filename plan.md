# Detailed, production-ready architecture and training plan (GPU & CPU)

Below is a comprehensive, step-by-step technical specification for implementing the multi-stage pipeline I recommended earlier — tuned for **Azerbaijani handwritten documents** and prepared both for **GPU training** (fast, preferred) and **CPU-only training/inference** (resource constrained). This document covers model architectures, exact layer choices and sizes, preprocessing, augmentation, training recipes, hyperparameters, optimization tricks, evaluation, checkpointing, monitoring, and CPU/GPU-specific deployment & inference optimizations.

---

## 1. High-level pipeline (reminder)

1. **Layout & region detection** (page → boxes: lines, fields, tables)
2. **HTR** (crop → text) — core recognizer: **CNN encoder → Transformer encoder → CTC decoder**
3. **Layout-aware semantic extraction** (recognized text + coords → structured fields; LayoutLM-style)
4. **Postprocessing & validation** (LM rescoring, normalization, domain rules)

This doc focuses mainly on Stage 2 (HTR) and the parts that interact with it (stages 1 & 3), with full instructions for both GPU and CPU training/inference.

---

# 2. Data & preprocessing (universal)

**2.1. Dataset structure**

* `images/` — page/line/word images (store original full-page and crops)
* `labels/` — plain text files with same basename or CSV: `image_path, transcription, doc_id, bbox, language_tag`
* `splits/` — `train.txt`, `val.txt`, `test.txt` (each file contains paths). **Split by document** (document-wise split) to avoid leakage.

**2.2. Input types**

* Use **line-level** crops as primary training input. Word-level crops for fine-grain models and full-page only as source for segmentation/detection training.

**2.3. Image preprocessing pipeline**

* Convert to grayscale (keep 3-channel optional for pretrained encoders).
* Deskew: compute image moments and rotate to zero skew; fallback: Hough or projection-profile method.
* Denoise: median / bilateral filter if heavy noise.
* Contrast normalize: adaptive histogram equalization (CLAHE) or contrast stretching.
* Resize: **height normalize** keeping aspect ratio. Typical heights: `imgH = 32` (fast), `imgH = 48` (better for longer sequences), `imgH = 64` (best quality). Use padding to maximum width in batch or use bucketing.
* Binarization only for very noisy scans — often losing grayscale info hurts HTR.

**2.4. Collapsing height → time steps (feature map time dimension)**

* Design CNN so that final feature map has shape `[B, C, H', W']` with `H' = 1` (or close), then reshape to `[B, T=W', C]`. Achieve via conv blocks with strides/pooling that reduce H dimension aggressively but preserve W. Example conv strides: along height: reduce factor 8 or 16; along width: keep stride 1 until final pooling.

**2.5. Data augmentation (critical for low-resource)**
Use `albumentations` or custom transforms. Probabilities and ranges:

* Rotation: `±3°` (p=0.5)
* Scale: `0.9–1.1` (p=0.5)
* Translation: `±2%` of height/width (p=0.3)
* Elastic distortions: alpha 30–36, sigma 5–6 (p=0.3)
* Brightness/contrast: `±30%` (p=0.5)
* Gaussian blur: sigma `0.5–1.5` (p=0.2)
* Gaussian noise: var up to `0.01` (p=0.3)
* Random erasing: area `0.01–0.05` (p=0.2)
* Inking bleed / stains (synthetic overlay): (p=0.15)
* Synthetic font-based generation: generate lines with Azerbaijani text using handwriting fonts and style-transfer → add to training (if permitted).

**2.6. Tokenization & vocabulary**

* **Character-level** vocabulary. Include Azerbaijani chars: `a-z`, `A-Z` (if mixed case), `ə ç ğ ı ö ş ü`, digits, punctuation `. , : ; - / ( ) % + =`, space, special tokens (CTC blank implicit).
* Map characters to integer labels; keep a `vocab.json`. For seq2seq experiments, prefer SentencePiece trained on Azerbaijani corpora (char-level or small BPE size 500–1000).

---

# 3. HTR architecture (detailed)

We choose  **CNN encoder → Transformer encoder → CTC decoder** . This gives the robustness of CNN features + long-range context via Transformer, while preserving CTC alignment simplicity.

**3.1. Exact layer-by-layer specification (recommended starting model: "HTR-Base")**

**Input:** grayscale image `H x W` (H chosen = 48 or 64), batch size variable.

**CNN Encoder (feature extractor)**

* ConvBlock1: `Conv2d(1, 64, kernel=3, stride=1, padding=1)`, `BatchNorm2d`, `ReLU`
* ConvBlock2: `Conv2d(64, 128, 3, stride=(2,1), padding=1)` → reduces height by 2, `BN`, `ReLU`
* ConvBlock3: `Conv2d(128, 256, 3, stride=1, padding=1)`, `BN`, `ReLU`
* ConvBlock4: `Conv2d(256, 256, 3, stride=(2,1), padding=1)` → reduce height by 2 again, `BN`, `ReLU`
* ConvBlock5: `Conv2d(256, 512, 3, stride=1, padding=1)`, `BN`, `ReLU`
* ConvBlock6: `Conv2d(512, 512, 3, stride=1, padding=1)`, `BN`, `ReLU`
* Optionally: `Conv2d(512, 512, 3, stride=(2,1), padding=1)` to further reduce height as needed.

Goal: final H' = 1 or 2; W' = time steps.

**Feature collapse to sequence**

* After final conv, apply `AdaptiveAvgPool2d((1, None))` or squeeze the height dimension -> get `[B, C, T]` then transpose -> `[B, T, C]`.

**Positional encoding**

* Add fixed sinusoidal or learned positional embedding of size `C`.

**Transformer Encoder**

* `num_layers = 6` (use 3 for lightweight)
* `d_model = 512` (match CNN output channels)
* `num_heads = 8`
* `dim_feedforward = 2048`
* PreNorm / PostNorm whichever stable; add dropout `0.1`

**Projection to vocab (CTC head)**

* `Linear(d_model, num_classes + 1)` (the +1 is CTC blank)
* LogSoftmax for CTC loss input.

**3.2. Variants**

* **HTR-Lite** : use fewer transformer layers (2–3), `d_model=256` — for GPU memory constrained or CPU training.
* **HTR-Heavy** : more transformer layers (8–12) and `d_model=768` for higher accuracy if GPU & data are sufficient.

**3.3. Model-size summary**

* HTR-Base (d_model=512, 6 layers) parameter count ~20–40M depending on CNN configuration.
* HTR-Lite ~6–12M — good for CPU training & deployment.

---

# 4. Training recipes

Two tracks: **GPU-optimized** (fast, use mixed precision, bigger batches) and **CPU-optimized** (smaller models, gradient accumulation, optimized dataloaders). Both contain fully reproducible commands.

### Shared hyperparameters (start point)

* Optimizer: `AdamW`
* Weight decay: `1e-4`
* Base LR (GPU): `1e-4` (with warmup); (CPU smaller model) `3e-4` might also work for small models — tune.
* Scheduler: `OneCycle` or `CosineAnnealingWarmRestarts` or `ReduceLROnPlateau` (use best checkpointing on val CER)
* Batch size per GPU: `32` for small H=48 or `16` for H=64 and heavy models.
* Epochs: `40–120` with early stopping on val CER (patience=10).
* Gradient clipping: `max_norm=1.0`
* Label smoothing: not used for CTC (no teacher-forcing).
* Seed: `seed=42`, set `torch.backends.cudnn.deterministic=False` for speed, but set seeds for reproducibility; log RNG states.

---

## 4.1. GPU training (preferred)

**4.1.1. Environment**

* PyTorch `>=1.12` with CUDA; install `apex` optional for fused optimizers (if available).
* Use `torch.cuda.amp` for mixed precision (autocast + GradScaler) to speed training & reduce memory.
* Use `torchrun --nproc_per_node=N` for multi-GPU distributed data parallel (DDP).

**4.1.2. Command line (single GPU)**

```bash
python train_htr.py \
  --config configs/htr_base.yaml \
  --device cuda:0 \
  --batch_size 32 \
  --imgH 48 \
  --epochs 80 \
  --amp True \
  --save_dir runs/htr_base
```

**4.1.3. Command line (multi-GPU, 4 GPUs)**

```bash
torchrun --nproc_per_node=4 train_htr.py --config configs/htr_base.yaml --batch_size 32 --amp True --save_dir runs/htr_base
```

Note: The `batch_size` is per-process. Total effective batch size = `batch_size * num_gpus`.

**4.1.4. Training loop highlights (pseudocode)**

* Use `DistributedSampler` for DDP.
* Wrap forward/backward in `autocast()` and scale grads with `GradScaler`.
* After each epoch evaluate on val set computing CER; save best checkpoint by val CER.
* Optionally validate after every N steps early in training.

**4.1.5. Example optimizer & scheduler**

* `optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)`
* Warmup: linear warmup for `warmup_steps = 2000` or epochs 1–2.
* Scheduler: `CosineAnnealingLR` or `OneCycleLR(max_lr=1e-3, total_steps=...)` for faster convergence.

**4.1.6. Mixed precision tips**

* Use `torch.cuda.amp.autocast()` for forward pass.
* Use `GradScaler()` to scale loss and `scaler.step(optimizer)`.

**4.1.7. Batch & training time expectations**

* HTR-Base on single RTX 2080 Ti: `batch=32` -> ~1.5–3K images per minute depending on image widths & augmentations. Full training ~ several hours to a day. Adjust batch and accumulation for memory.

---

## 4.2. CPU training (constrained)

When only CPU is available, adapt model and pipeline:

**4.2.1. Strategy**

* Use **HTR-Lite** (d_model=256, 2–3 transformer layers)
* Use efficient PyTorch settings: `torch.set_num_threads(num_cores)` to parallelize CPU ops. Prefer `num_workers=4-8` for DataLoader.
* Use **gradient accumulation** to simulate larger batch size: e.g., `batch_size=8`, `accumulate_steps=4` → effective batch=32.
* Use mixed precision? Not applicable on CPU (PyTorch supports `bfloat16` on some CPUs but rarely).
* Use `torch.compile()` (PyTorch 2.0) if available to speed up.
* Use Intel MKL, OpenMP optimized PyTorch builds or `oneAPI` + OpenVINO for better performance.

**4.2.2. Command example**

```bash
python train_htr.py \
  --config configs/htr_lite_cpu.yaml \
  --device cpu \
  --batch_size 8 \
  --accumulate_steps 4 \
  --imgH 48 \
  --epochs 120 \
  --num_workers 8 \
  --save_dir runs/htr_cpu
```

**4.2.3. Hyperparameters changes**

* LR: slightly higher `2e-4` for small models, with careful warmup.
* Epochs: more epochs (80–150) because CPU training is slower; early stopping must still be used.
* Augmentation: still use but reduce heavy transforms that are CPU-expensive or precompute augmented dataset offline.

**4.2.4. Data pipeline optimizations**

* Use cached images in a memory-mapped format (LMDB) or pre-resized PNG/NPY to reduce on-the-fly cost.
* Precompute augmentations offline or use lower augmentation probability during CPU training.

**4.2.5. Expected time**

* CPU training is significantly slower; small model may take multiple days. Consider hybrid approach: do model development on GPU (cloud/hybrid) and only run final small model training locally on CPU.

---

# 5. Language Model rescoring (beam search + char LM)

**5.1. Why**
A small char/subword LM trained on Azerbaijani text (news, web, Wikipedia) will dramatically reduce CER by preferring plausible character sequences during beam search.

**5.2. Build LM**

* Use `kenlm` to build an n-gram char LM (3-gram or 5-gram).
* Train on a cleaned Azerbaijani corpus; preserve diacritics.

Command example:

```bash
# prepare corpus: each line plain text
kenlm/bin/lmplz -o 5 < az_corpus.txt > az_char5.arpa
kenlm/bin/build_binary az_char5.arpa az_char5.binary
```

**5.3. Beam search with LM**

* Use beam search width `beam=10` (tune 5–20).
* Score = `CTC_logprob + alpha * LM_logprob + beta * word_count_penalty`
* Tune `alpha` (LM weight) and `beta` on validation set (grid search: alpha 0.1–2.0).

**5.4. Implementation**

* Use `ctcdecode` package or a custom implementation that takes KenLM probabilities.

---

# 6. Evaluation & metrics

**6.1. Primary metrics**

* **CER** (character error rate) — primary.
* **WER** (word error rate).
* **Exact match** (for field extraction).
* **Field-level F1** for structured extraction.

**6.2. How to compute CER/WER**

* Use Levenshtein distance between predicted vs reference. Normalize tokens (lowercase, unify diacritics choices if needed). Keep diacritics in primary eval.

**6.3. Error analysis**

* Produce per-character confusion matrix.
* Stratify results by writer, document type, line length, image quality.
* Keep per-bbox confidence to find low-confidence zones.

---

# 7. Checkpoints, reproducibility, & monitoring

**7.1. Checkpointing**

* Save model checkpoint every N epochs/steps and keep best by `val_cer`.
* Save optimizer & scheduler state for resume.

**7.2. Reproducibility**

* Log random seeds for Python, NumPy, PyTorch.
* Save `vocab.json`, `config.yaml`, dataset split files.

**7.3. Monitoring**

* Use TensorBoard or Weights & Biases (W&B). Log: training loss, val CER/WER, learning rate, gradient norms, sample predictions, and confusion analysis images.

---

# 8. Inference & deployment (CPU-focused optimizations)

**8.1. Offline inference**

* For production on CPU, convert trained PyTorch model to **ONNX** and run with **onnxruntime** or **OpenVINO** for Intel CPUs.

**8.2. Quantization**

* **Dynamic quantization** (PyTorch) for linear layers can reduce model size and increase CPU throughput with minimal accuracy loss. Do:

```python
quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

* For more aggressive gains, use **Quantization Aware Training (QAT)** — usually requires GPU for training.

**8.3. Pruning & distillation**

* Use teacher-student distillation to create smaller student model. Distill transformer encoder outputs to a smaller model trained with a distillation loss (KL between logits).
* Structured pruning (channels/attention heads) can be applied post-training.

**8.4. ONNX export**

* Export with dynamic axes for variable width/time steps. Example:

```python
torch.onnx.export(model, sample_input, "htr.onnx",
                  input_names=["image"], output_names=["logits"],
                  dynamic_axes={"image": {3: "width"}, "logits": {1: "time"}})
```

**8.5. Inference pipeline**

* Page → detection → crop → preprocess → model (ONNX) → CTC decode → LM rescoring → layout-aware extractor → postprocess.

**8.6. Latency targets**

* HTR-Lite: aim for `<=300 ms` per line on a modern CPU with quantized ONNX.
* HTR-Base (heavy): `100–200 ms` per line on GPU.

---

# 9. Layout detection & semantic extraction (integration details)

**9.1. Layout detection**

* Train object detector (YOLOv8 / Faster-RCNN / Detectron2) to output text lines, fields, stamps. Use small backbone (CSPDarkNet, ResNet50) depending on GPU.
* Loss: typical object detection losses; add IoU threshold tuning.

**9.2. Semantic extraction**

* Use **LayoutLMv3-style** model that accepts recognized text + bounding boxes (x,y,w,h normalized) to classify tokens into fields or extract key-value pairs. Fine-tune multilingual pre-trained LayoutLM or XLM-R with Azerbaijani data (or use multilingual tokenizers).
* If data for fine-tuning is scarce, use a rule-based parser with regex + heuristics as fallback.

---

# 10. Training & infra scripts (what to include in repo)

**10.1. Key files**

* `train_htr.py` — training entrypoint
* `models/htr.py` — model definitions (HTR-Base, HTR-Lite)
* `configs/*.yaml` — all experiment configs
* `data_loader.py` — DataLoader with augmentations & bucketing
* `infer.py` — inference script with beam + LM rescoring
* `utils/metrics.py` — CER/WER calculators
* `requirements.txt` — pinned deps
* `Dockerfile` — reproducible environment for inference/demo

**10.2. Example `requirements.txt` (minimal)**

```
torch>=1.12
torchvision
albumentations
opencv-python
numpy
tqdm
python-Levenshtein
ctcdecode
kenlm
onnxruntime
onnx
tensorboard
gradio
detectron2 (if used)
transformers
sentencepiece
```

**10.3. Example Dockerfile (inference)**

```dockerfile
FROM python:3.10-slim
RUN apt-get update && apt-get install -y build-essential libsndfile1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
CMD ["python", "infer.py", "--model", "htr.onnx", "--input", "demo.png"]
```

---

# 11. Practical tuning checklist (quick actionable)

1. **Start with HTR-Lite + heavy augmentation** to get a working baseline in a few hours.
2. **Train char-level KenLM** concurrently.
3. **Add Transformer encoder** when baseline plateaus.
4. **Use mixed precision & DDP** to speed up GPU experiments.
5. **Tune LM weight (`alpha`) on validation** grid: `[0.1,0.2,0.5,1.0,1.5]`.
6. **Perform pseudo-labeling** on unlabeled pages: infer with high threshold, add to train, retrain.
7. **Create active learning loop** : label high-uncertainty samples first.
8. **Quantize and export** ONNX for CPU deploy; measure throughput & CER trade-off.

---

# 12. Example hyperparameter sets

**HTR-Base (GPU)**

* imgH = 48, d_model=512, n_layers=6, n_heads=8
* batch_size = 32 (per-GPU)
* lr = 1e-4 (AdamW), warmup 2k steps
* epochs = 60–80
* beam = 10, LM weight alpha=0.8

**HTR-Lite (CPU)**

* imgH = 48, d_model=256, n_layers=3, n_heads=4
* batch_size = 8, accumulate_steps = 4
* lr = 2e-4, warmup 1k steps
* epochs = 80–120
* beam = 5, LM weight alpha=0.5

---

# 13. Example training snippet (PyTorch pseudocode)

```python
# core loop simplified
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
for epoch in range(start_epoch, epochs):
    model.train()
    for step, batch in enumerate(train_loader):
        imgs, targets, lengths = batch
        imgs = imgs.to(device)
        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            logits = model(imgs)  # [B, T, C]
            log_probs = F.log_softmax(logits, dim=-1).transpose(0,1)  # T,B,C for CTCLoss
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    # validate, save best
```

---

# 14. Risk & failure modes and mitigations

* **Overfitting to small writers:** use heavy augmentation, synthetic data, and writer-split cross-validation.
* **Data leakage via doc split misuse:** ensure doc-based split.
* **LM bias:** LM can force wrong diacritics -> tune LM weight and use character LM.
* **Slow CPU inference:** use quantization + ONNX + batching.

---

# 15. Final notes (Azerbaijani specifics)

* Preserve diacritics during all preprocessing and LM corpora. Evaluate with diacritics on.
* Build small Azerbaijani corpus early; even 100k–1M tokens gives LM boost. Use news articles, open corpora.
* Pay attention to names and domain lexicons (industry words for SOCAR). Add them to rescoring lexicon or use constrained decoding for fields (e.g., ID patterns).

---
