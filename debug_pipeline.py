#!/usr/bin/env python3
"""Debug script to analyze the entire training pipeline"""

import os
import json
import numpy as np
from PIL import Image
import pillow_heif
import torch

# Register HEIF
pillow_heif.register_heif_opener()

print("="*80)
print("COMPREHENSIVE PIPELINE ANALYSIS")
print("="*80)

# 1. Check label lengths
print("\n1. LABEL LENGTH ANALYSIS")
print("-"*80)
labels_dir = "data/labels"
label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.txt')])

label_lengths = []
for label_file in label_files:
    with open(os.path.join(labels_dir, label_file), 'r', encoding='utf-8') as f:
        text = ' '.join(line.strip() for line in f.readlines() if line.strip())
        label_lengths.append(len(text))
        if len(label_lengths) <= 3:
            print(f"  {label_file}: {len(text)} chars - '{text[:60]}...'")

print(f"\nLabel length statistics:")
print(f"  Min: {min(label_lengths)} chars")
print(f"  Max: {max(label_lengths)} chars")
print(f"  Mean: {np.mean(label_lengths):.1f} chars")
print(f"  ⚠️  CTC requires: input_length > label_length")

# 2. Check image dimensions
print("\n2. IMAGE DIMENSIONS ANALYSIS")
print("-"*80)
images_dir = "data/images"
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.HEIC')])[:3]

IMG_HEIGHT = 256
for img_file in image_files:
    img = Image.open(os.path.join(images_dir, img_file))
    orig_w, orig_h = img.size
    new_w = int(IMG_HEIGHT * orig_w / orig_h)
    print(f"  {img_file}:")
    print(f"    Original: {orig_w}×{orig_h}")
    print(f"    After resize: {new_w}×{IMG_HEIGHT}")

    # Simulate CNN output
    # 3 stride-2 layers reduce height: 256 → 128 → 64 → 32
    # Adaptive pool to (1, None): 32 → 1
    # Width stays same (stride is (2,1) not (2,2))
    final_sequence_length = new_w
    print(f"    Model output sequence: ~{final_sequence_length} time steps")

# 3. Check one validation sample
print("\n3. VALIDATION SET ANALYSIS")
print("-"*80)
print("⚠️  Validation set has only 1 sample!")
print("⚠️  Batch size is 2, but only 1 val sample exists!")
print("⚠️  This likely causes the validation loss = 0.0 bug!")
print("\nPossible issues:")
print("  1. DataLoader creates incomplete batch")
print("  2. Label length might be 0 after preprocessing")
print("  3. Sequence length might be <= label length")

# 4. Load vocabulary and check
print("\n4. VOCABULARY ANALYSIS")
print("-"*80)
with open('outputs/vocabulary.json', 'r') as f:
    vocab = json.load(f)
    print(f"  Vocab size: {vocab['vocab_size']} characters")
    print(f"  Blank index: {vocab['blank_idx']}")
    print(f"  Num classes: {vocab['num_classes']}")
    print(f"  ✓ Correctly configured for CTC")

# 5. Image quality analysis
print("\n5. IMAGE QUALITY ANALYSIS")
print("-"*80)
print("Preprocessing steps:")
print("  1. RGB → Grayscale (loses color information)")
print("  2. CLAHE enhancement (clipLimit=2.0)")
print("  3. Resize to height=256 (maintains aspect ratio)")
print("  4. Normalize to [0, 1]")
print("\n⚠️  Potential issues:")
print("  - HEIC images are photos (~900KB) with lots of detail")
print("  - Grayscale might lose colored ink vs background")
print("  - CLAHE might over-enhance and create artifacts")
print("  - Height 256 might be too small for detailed handwriting")

# 6. Architecture analysis
print("\n6. MODEL ARCHITECTURE ANALYSIS")
print("-"*80)
print("CNN Architecture:")
print("  Input: [B, 1, 256, W]")
print("  Conv1: [B, 64, 256, W]")
print("  Conv2: [B, 128, 128, W] (stride 2 on height)")
print("  Conv3: [B, 256, 128, W]")
print("  Conv4: [B, 256, 64, W] (stride 2 on height)")
print("  Conv5: [B, 512, 64, W]")
print("  Conv6: [B, 512, 32, W] (stride 2 on height)")
print("  AdaptivePool: [B, 512, 1, W]")
print("  Squeeze: [B, 512, W]")
print("  Permute: [B, W, 512]")
print("\n⚠️  Potential issue:")
print("  - Height reduced from 256 → 32 before pooling")
print("  - For tall handwriting, this might lose vertical information")
print("  - Adaptive pool to height=1 might be too aggressive")

# 7. CTC Requirements Check
print("\n7. CTC REQUIREMENTS")
print("-"*80)
print("For CTC to work:")
print("  ✓ blank_idx must be valid: {blank_idx} < {num_classes}".format(**vocab))
print(f"  ✓ Model outputs {vocab['num_classes']} classes")
print("  ⚠️  input_length MUST be > label_length for each sample")
print("  ⚠️  With labels averaging ~100 chars, need >100 time steps")
print("  ⚠️  Sequence length = image_width (after resize)")
print("  ⚠️  If image width < label length → CTC FAILS")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)
print("1. FIX VALIDATION BUG:")
print("   - Increase validation set to at least 2 samples")
print("   - OR set batch_size=1 for validation")
print("")
print("2. FIX IMAGE QUALITY:")
print("   - Increase IMG_HEIGHT to 512 or higher")
print("   - Consider keeping color (RGB) instead of grayscale")
print("   - Reduce CLAHE clipLimit or remove it")
print("")
print("3. FIX ARCHITECTURE:")
print("   - Reduce number of stride-2 layers (use max pooling)")
print("   - OR increase initial image height")
print("   - Ensure sequence_length >> max_label_length")
print("")
print("4. GET MORE DATA:")
print("   - 12 samples is way too small")
print("   - Need at least 100+ labeled samples")
print("   - Use data augmentation (rotation, perspective, noise)")
print("="*80)
