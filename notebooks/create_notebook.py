#!/usr/bin/env python3
"""Create training notebook with image preprocessing visualization"""
import nbformat as nbf

nb = nbf.v4.new_notebook()

# Cell 0: Title
nb.cells.append(nbf.v4.new_markdown_cell("""# Azeri Handwriting Recognition - HTR Training

Based on the architecture in `plan.md`:
- CNN Feature Extractor → Transformer Encoder → CTC Decoder
- IMG_HEIGHT = 256 (readable text)
- Preprocessing visualization at each stage
- Full page recognition

**Note:** Run all cells to train the model. Images at each preprocessing stage will be displayed."""))

# Cell 1: Imports
nb.cells.append(nbf.v4.new_code_cell("""# All imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import cv2
from pathlib import Path
import albumentations as A
from PIL import Image
from pillow_heif import register_heif_opener
import matplotlib.pyplot as plt
import pandas as pd
import json
import random
from tqdm import tqdm
import Levenshtein
from collections import defaultdict

# Register HEIF support
register_heif_opener()

print("✓ All imports successful!")"""))

# Cell 2: Config
nb.cells.append(nbf.v4.new_code_cell("""# Configuration - HTR Architecture from plan.md
class Config:
    PROJECT_ROOT = Path('/Users/ismatsamadov/azeri_handwriting_detection')
    DATA_DIR = PROJECT_ROOT / 'data'
    IMAGES_DIR = DATA_DIR / 'images'
    LABELS_DIR = DATA_DIR / 'labels'
    CHARTS_DIR = PROJECT_ROOT / 'charts'
    MODEL_DIR = PROJECT_ROOT / 'model_artifacts'

    # Image preprocessing
    IMG_HEIGHT = 256  # Readable text size

    # HTR Architecture (from plan.md)
    CNN_CHANNELS = [32, 64, 128, 256]  # Feature extraction
    D_MODEL = 256  # Transformer dimension
    N_HEADS = 8  # Attention heads
    N_LAYERS = 4  # Transformer layers
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1

    # Training
    BATCH_SIZE = 2
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    GRADIENT_CLIP = 1.0

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MIXED_PRECISION = torch.cuda.is_available()

    TRAIN_DOCS = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
    VAL_DOCS = ['10', '11']
    TEST_DOCS = ['12']

    SEED = 42

    @classmethod
    def create_dirs(cls):
        cls.CHARTS_DIR.mkdir(exist_ok=True)
        cls.MODEL_DIR.mkdir(exist_ok=True)
        (cls.CHARTS_DIR / 'preprocessing').mkdir(exist_ok=True)

Config.create_dirs()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(Config.SEED)
print(f"Device: {Config.DEVICE}")
print(f"IMG_HEIGHT: {Config.IMG_HEIGHT}px (maintains aspect ratio)")"""))

# Cell 3: Preprocessing with visualization
nb.cells.append(nbf.v4.new_code_cell("""# Image Preprocessing with Stage-by-Stage Visualization
class ImagePreprocessor:
    def __init__(self, target_height: int = 256, visualize=False):
        self.target_height = target_height
        self.visualize = visualize

    def process(self, image: np.ndarray, img_name="sample"):
        stages = {}

        # Stage 0: Original
        stages['0_original'] = image.copy()
        orig_h, orig_w = image.shape[:2] if len(image.shape) == 2 else image.shape[:2]

        # Stage 1: Grayscale conversion
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        stages['1_grayscale'] = image.copy()

        # Stage 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        stages['2_clahe'] = image.copy()

        # Stage 3: Resize (maintain aspect ratio)
        h, w = image.shape
        new_width = int(self.target_height * w / h)
        image = cv2.resize(image, (new_width, self.target_height),
                          interpolation=cv2.INTER_CUBIC)
        stages['3_resized'] = image.copy()

        print(f"✓ Preprocessing: {orig_h}×{orig_w} → {self.target_height}×{new_width}")

        # Stage 4: Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        stages['4_normalized'] = image.copy()

        # Visualize if requested
        if self.visualize:
            self._visualize_stages(stages, img_name)

        return image, stages

    def _visualize_stages(self, stages, img_name):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        stage_names = [
            '0: Original',
            '1: Grayscale',
            '2: CLAHE (Contrast)',
            '3: Resized',
            '4: Normalized',
            '5: Info'
        ]

        for idx, (key, img) in enumerate(stages.items()):
            if idx < 5:
                axes[idx].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
                axes[idx].set_title(f"{stage_names[idx]}\\n{img.shape}", fontsize=10)
                axes[idx].axis('off')

        # Info panel
        axes[5].axis('off')
        info_text = f"""Image: {img_name}

Original size: {stages['0_original'].shape}
Grayscale: {stages['1_grayscale'].shape}
After CLAHE: {stages['2_clahe'].shape}
Resized: {stages['3_resized'].shape}
Normalized: {stages['4_normalized'].shape}

Target height: {self.target_height}px
Aspect ratio: preserved"""

        axes[5].text(0.1, 0.5, info_text, fontsize=11, family='monospace',
                    verticalalignment='center')

        plt.tight_layout()
        plt.savefig(Config.CHARTS_DIR / 'preprocessing' / f'{img_name}_stages.png',
                   dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Saved visualization: {img_name}_stages.png")

preprocessor = ImagePreprocessor(Config.IMG_HEIGHT, visualize=True)
print(f"✓ Preprocessor ready (IMG_HEIGHT={Config.IMG_HEIGHT}, visualization=ON)")"""))

# Cell 4: Test preprocessing on first image
nb.cells.append(nbf.v4.new_code_cell("""# Test preprocessing on first image to see transformation
print("="*80)
print("PREPROCESSING VISUALIZATION TEST")
print("="*80)

# Load first image
first_img_path = list(Config.IMAGES_DIR.glob("*.HEIC"))[0]
print(f"Loading: {first_img_path.name}")

img_pil = Image.open(first_img_path)
img_np = np.array(img_pil)

print(f"Original image shape: {img_np.shape}")
print(f"Original dtype: {img_np.dtype}")
print()

# Process with visualization
processed_img, stages = preprocessor.process(img_np, img_name=first_img_path.stem)

print()
print("Stage shapes:")
for name, img in stages.items():
    print(f"  {name}: {img.shape} | dtype: {img.dtype} | range: [{img.min():.2f}, {img.max():.2f}]")

print()
print("✓ Check the visualization above to see each preprocessing stage")
print("="*80)"""))

# Cell 5: Dataset
nb.cells.append(nbf.v4.new_code_cell("""# Dataset class
class HTRDataset(Dataset):
    def __init__(self, doc_ids, preprocessor, images_dir, labels_dir, augmentation=None):
        self.doc_ids = doc_ids
        self.preprocessor = preprocessor
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.augmentation = augmentation
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for doc_id in self.doc_ids:
            img_path = self.images_dir / f"{doc_id}.HEIC"

            # Find label file
            label_file = None
            for lf in self.labels_dir.glob(f'*_{doc_id}.txt'):
                label_file = lf
                break

            if not label_file or not label_file.exists():
                continue

            # Read label file directly (plain text, no arrow delimiter)
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                full_text = ' '.join(lines)

            if full_text:
                samples.append((img_path, full_text))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]

        # Load image
        image = Image.open(img_path)
        image = np.array(image)

        # Preprocess (without visualization for dataset)
        proc = ImagePreprocessor(self.preprocessor.target_height, visualize=False)
        image, _ = proc.process(image, img_path.stem)

        # Augmentation
        if self.augmentation:
            augmented = self.augmentation(image=image)
            image = augmented['image']

        return {
            'image': image,
            'text': text,
            'img_path': str(img_path)
        }

print("✓ Dataset class defined")"""))

# Cell 6: Build vocabulary
nb.cells.append(nbf.v4.new_code_cell("""# Build vocabulary from all labels
all_chars = set()
for doc_id in Config.TRAIN_DOCS + Config.VAL_DOCS + Config.TEST_DOCS:
    for lf in Config.LABELS_DIR.glob(f'*_{doc_id}.txt'):
        with open(lf, 'r', encoding='utf-8') as f:
            text = f.read()
            all_chars.update(text)

sorted_chars = sorted(all_chars)
char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted_chars)}  # 0 for CTC blank
char_to_idx['<BLANK>'] = 0
idx_to_char = {idx: char for char, idx in char_to_idx.items()}
vocab_size = len(char_to_idx)

print(f"Vocabulary: {vocab_size} characters")
print(f"  Blank token: 0")
print(f"  Character range: {min(char_to_idx.values())} - {max(char_to_idx.values())}")
print(f"  Sample chars: {sorted_chars[:20]}")

# Save vocabulary
with open(Config.MODEL_DIR / 'vocabulary.json', 'w', encoding='utf-8') as f:
    json.dump(char_to_idx, f, ensure_ascii=False, indent=2)
print(f"\\n✓ Saved vocabulary to {Config.MODEL_DIR / 'vocabulary.json'}")"""))

# Cell 7: Data loaders
nb.cells.append(nbf.v4.new_code_cell("""# Data loaders with collate function
def get_augmentation():
    return A.Compose([
        A.Rotate(limit=3, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=1.0),
        A.Affine(scale=(0.95, 1.05), p=0.5, mode=cv2.BORDER_CONSTANT, cval=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.GaussNoise(var_limit=(0.001, 0.005), p=0.3),
    ])

def collate_fn(batch):
    images = [item['image'] for item in batch]
    texts = [item['text'] for item in batch]

    # Pad images to same width
    max_width = max(img.shape[1] for img in images)
    padded_images = []
    for img in images:
        h, w = img.shape
        padded = np.ones((h, max_width), dtype=np.float32)
        padded[:, :w] = img
        padded_images.append(padded)

    # Convert to tensors
    images_tensor = torch.FloatTensor(np.array(padded_images)).unsqueeze(1)  # [B, 1, H, W]

    # Encode texts
    labels = []
    label_lengths = []
    for text in texts:
        encoded = [char_to_idx.get(c, 0) for c in text]
        labels.extend(encoded)
        label_lengths.append(len(encoded))

    labels = torch.LongTensor(labels)
    label_lengths = torch.LongTensor(label_lengths)

    return {
        'images': images_tensor,
        'labels': labels,
        'label_lengths': label_lengths,
        'label_texts': texts
    }

# Create datasets
train_dataset = HTRDataset(
    Config.TRAIN_DOCS, preprocessor, Config.IMAGES_DIR,
    Config.LABELS_DIR, augmentation=get_augmentation()
)
val_dataset = HTRDataset(
    Config.VAL_DOCS, preprocessor, Config.IMAGES_DIR, Config.LABELS_DIR
)
test_dataset = HTRDataset(
    Config.TEST_DOCS, preprocessor, Config.IMAGES_DIR, Config.LABELS_DIR
)

train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE,
                          shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE,
                        shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE,
                         shuffle=False, collate_fn=collate_fn)

print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
print(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")
print(f"Test: {len(test_dataset)} samples, {len(test_loader)} batches")

# Test batch
sample = next(iter(train_loader))
print(f"\\nSample batch:")
print(f"  Images: {sample['images'].shape}")
print(f"  Labels: {sample['labels'].shape}")
print(f"  Label lengths: {sample['label_lengths']}")
print(f"  Texts: {len(sample['label_texts'])} texts")"""))

# Cell 8: HTR Model
nb.cells.append(nbf.v4.new_code_cell("""# HTR Model - Architecture from plan.md
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class HTRModel(nn.Module):
    \"\"\"HTR Architecture: CNN → Transformer → CTC\"\"\"
    def __init__(self, vocab_size, img_height=256,
                 cnn_channels=[32, 64, 128, 256],
                 d_model=256, n_heads=8, n_layers=4,
                 dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # CNN Feature Extractor
        layers = []
        in_ch = 1
        for idx, out_ch in enumerate(cnn_channels):
            # Stride (2,1) every other layer to reduce height
            stride = (2, 1) if idx % 2 == 1 else (1, 1)
            layers.extend([
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ])
            in_ch = out_ch
        self.cnn_encoder = nn.Sequential(*layers)

        # Adaptive pooling to collapse height
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))

        # Project to d_model if needed
        self.feature_projection = nn.Linear(cnn_channels[-1], d_model) if cnn_channels[-1] != d_model else nn.Identity()

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        # CTC Head
        self.ctc_head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # CNN: [B, 1, H, W] → [B, C, H', W']
        features = self.cnn_encoder(x)

        # Pool height: [B, C, H', W'] → [B, C, 1, W']
        features = self.adaptive_pool(features).squeeze(2)

        # Transpose: [B, C, W'] → [W', B, C]
        features = features.permute(2, 0, 1)

        # Project: [W', B, C] → [W', B, d_model]
        features = self.feature_projection(features)

        # Add positional encoding
        features = self.pos_encoder(features)

        # Transformer: [W', B, d_model] → [W', B, d_model]
        transformer_out = self.transformer_encoder(features)

        # CTC: [W', B, d_model] → [W', B, vocab_size]
        logits = self.ctc_head(transformer_out)

        # Transpose: [W', B, vocab_size] → [B, W', vocab_size]
        return logits.permute(1, 0, 2)

model = HTRModel(
    vocab_size=vocab_size,
    img_height=Config.IMG_HEIGHT,
    cnn_channels=Config.CNN_CHANNELS,
    d_model=Config.D_MODEL,
    n_heads=Config.N_HEADS,
    n_layers=Config.N_LAYERS,
    dim_feedforward=Config.DIM_FEEDFORWARD,
    dropout=Config.DROPOUT
).to(Config.DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"HTR Model: {total_params:,} parameters ({total_params/1e6:.2f}M)")
print(f"  CNN channels: {Config.CNN_CHANNELS}")
print(f"  Transformer: d_model={Config.D_MODEL}, heads={Config.N_HEADS}, layers={Config.N_LAYERS}")

# Test forward pass
with torch.no_grad():
    out = model(sample['images'].to(Config.DEVICE))
    print(f"\\nForward pass test:")
    print(f"  Input: {sample['images'].shape}")
    print(f"  Output: {out.shape} [B, T, vocab_size]")
    print(f"  Time steps (T): {out.shape[1]}")"""))

# Cell 9: Training utilities
nb.cells.append(nbf.v4.new_code_cell("""# Training utilities
def ctc_decode_greedy(logits, idx_to_char):
    predictions = torch.argmax(logits, dim=-1)
    decoded_texts = []
    for pred in predictions:
        chars = []
        prev_char = None
        for idx in pred.cpu().numpy():
            if idx != 0 and idx != prev_char:
                chars.append(idx_to_char.get(idx, ''))
            prev_char = idx
        decoded_texts.append(''.join(chars))
    return decoded_texts

def calculate_cer(predictions, targets):
    total_dist, total_len = 0, 0
    for pred, target in zip(predictions, targets):
        total_dist += Levenshtein.distance(pred, target)
        total_len += len(target)
    return total_dist / total_len if total_len > 0 else 0.0

def calculate_wer(predictions, targets):
    total_dist, total_words = 0, 0
    for pred, target in zip(predictions, targets):
        pred_words = pred.split()
        target_words = target.split()
        total_dist += Levenshtein.distance(' '.join(pred_words), ' '.join(target_words))
        total_words += len(target_words)
    return total_dist / total_words if total_words > 0 else 0.0

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

print("✓ Utilities defined")"""))

# Cell 10: Training setup
nb.cells.append(nbf.v4.new_code_cell("""# Training setup
criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE,
                              weight_decay=Config.WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=Config.LEARNING_RATE * 5, epochs=Config.EPOCHS,
    steps_per_epoch=len(train_loader), pct_start=0.1
)
scaler = torch.cuda.amp.GradScaler() if Config.MIXED_PRECISION else None
early_stopping = EarlyStopping(patience=10)
writer = SummaryWriter(log_dir=str(Config.MODEL_DIR / 'runs'))

history = {
    'train_loss': [], 'val_loss': [],
    'train_cer': [], 'val_cer': [], 'val_wer': [],
    'learning_rate': []
}

print(f"✓ Training setup complete")
print(f"  Optimizer: AdamW (lr={Config.LEARNING_RATE})")
print(f"  Scheduler: OneCycleLR")
print(f"  Epochs: {Config.EPOCHS}")
print(f"  Early stopping patience: 10")"""))

# Cell 11: Training functions
nb.cells.append(nbf.v4.new_code_cell("""# Training and validation functions
def train_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, epoch):
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]")
    for batch in pbar:
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)
        label_lengths = batch['label_lengths']
        label_texts = batch['label_texts']

        optimizer.zero_grad()

        if Config.MIXED_PRECISION:
            with torch.cuda.amp.autocast():
                logits = model(images)
                log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
                input_lengths = torch.full((log_probs.size(1),), log_probs.size(0), dtype=torch.long)
                loss = criterion(log_probs, labels, input_lengths, label_lengths)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
            input_lengths = torch.full((log_probs.size(1),), log_probs.size(0), dtype=torch.long)
            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)
            optimizer.step()

        scheduler.step()

        preds = ctc_decode_greedy(logits, idx_to_char)
        all_preds.extend(preds)
        all_targets.extend(label_texts)
        total_loss += loss.item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(loader), calculate_cer(all_preds, all_targets)

def validate_epoch(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Epoch {epoch+1} [Val]"):
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)
            label_lengths = batch['label_lengths']
            label_texts = batch['label_texts']

            logits = model(images)
            log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
            input_lengths = torch.full((log_probs.size(1),), log_probs.size(0), dtype=torch.long)
            loss = criterion(log_probs, labels, input_lengths, label_lengths)

            preds = ctc_decode_greedy(logits, idx_to_char)
            all_preds.extend(preds)
            all_targets.extend(label_texts)
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    cer = calculate_cer(all_preds, all_targets)
    wer = calculate_wer(all_preds, all_targets)

    return avg_loss, cer, wer, all_preds, all_targets

print("✓ Training functions ready")"""))

# Cell 12: Main training loop
nb.cells.append(nbf.v4.new_code_cell("""# Main training loop
print("="*80)
print("STARTING TRAINING")
print("="*80)

best_val_cer = float('inf')
best_epoch = 0

for epoch in range(Config.EPOCHS):
    train_loss, train_cer = train_epoch(
        model, train_loader, criterion, optimizer, scheduler, scaler, Config.DEVICE, epoch
    )
    val_loss, val_cer, val_wer, val_preds, val_targets = validate_epoch(
        model, val_loader, criterion, Config.DEVICE, epoch
    )

    current_lr = optimizer.param_groups[0]['lr']

    # Update history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_cer'].append(train_cer)
    history['val_cer'].append(val_cer)
    history['val_wer'].append(val_wer)
    history['learning_rate'].append(current_lr)

    # TensorBoard
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('CER/train', train_cer, epoch)
    writer.add_scalar('CER/val', val_cer, epoch)
    writer.add_scalar('WER/val', val_wer, epoch)
    writer.add_scalar('LR', current_lr, epoch)

    # Print progress
    print(f"\\nEpoch {epoch+1}/{Config.EPOCHS}:")
    print(f"  Train Loss: {train_loss:.4f} | CER: {train_cer:.4f}")
    print(f"  Val Loss: {val_loss:.4f} | CER: {val_cer:.4f} | WER: {val_wer:.4f}")
    print(f"  LR: {current_lr:.6f}")

    # Show sample predictions
    if val_preds:
        print(f"\\n  Sample Prediction:")
        print(f"    Target: '{val_targets[0][:80]}...'")
        print(f"    Pred:   '{val_preds[0][:80]}...'")

    # Save best model
    if val_cer < best_val_cer:
        best_val_cer = val_cer
        best_epoch = epoch

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_cer': val_cer,
            'val_wer': val_wer,
        }, Config.MODEL_DIR / 'best_model.pth')
        print(f"  ✓ Best model saved (CER: {val_cer:.4f})")

    # Early stopping
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"\\nEarly stopping at epoch {epoch+1}")
        break

    print("-" * 80)

writer.close()
print(f"\\n✓ Training complete! Best: Epoch {best_epoch+1}, CER: {best_val_cer:.4f}")"""))

# Cell 13: Visualizations
nb.cells.append(nbf.v4.new_code_cell("""# Create training visualizations
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Loss curves
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True)

# 2. CER curves
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(history['train_cer'], label='Train CER', linewidth=2)
ax2.plot(history['val_cer'], label='Val CER', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('CER')
ax2.set_title('Character Error Rate')
ax2.legend()
ax2.grid(True)

# 3. WER curve
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(history['val_wer'], label='Val WER', linewidth=2, color='green')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('WER')
ax3.set_title('Word Error Rate')
ax3.legend()
ax3.grid(True)

# 4. Learning rate
ax4 = fig.add_subplot(gs[1, 2])
ax4.plot(history['learning_rate'], linewidth=2, color='orange')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Learning Rate')
ax4.set_title('LR Schedule')
ax4.set_yscale('log')
ax4.grid(True)

# 5. Final metrics
ax5 = fig.add_subplot(gs[2, 0])
metrics = ['Train Loss', 'Val Loss', 'Train CER', 'Val CER', 'Val WER']
values = [history['train_loss'][-1], history['val_loss'][-1],
          history['train_cer'][-1], history['val_cer'][-1], history['val_wer'][-1]]
ax5.bar(metrics, values, color=['blue', 'orange', 'green', 'red', 'purple'])
ax5.set_ylabel('Value')
ax5.set_title('Final Epoch Metrics')
ax5.tick_params(axis='x', rotation=45)
for i, v in enumerate(values):
    ax5.text(i, v, f'{v:.3f}', ha='center', va='bottom')

# 6. Best vs Final
ax6 = fig.add_subplot(gs[2, 1])
comparison = ['Best CER', 'Final CER', 'Best WER', 'Final WER']
comp_values = [best_val_cer, history['val_cer'][-1],
               min(history['val_wer']), history['val_wer'][-1]]
ax6.bar(comparison, comp_values, color=['green', 'lightgreen', 'blue', 'lightblue'])
ax6.set_ylabel('Error Rate')
ax6.set_title('Best vs Final Performance')
ax6.tick_params(axis='x', rotation=45)
for i, v in enumerate(comp_values):
    ax6.text(i, v, f'{v:.3f}', ha='center', va='bottom')

# 7. Training info
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')
info_text = f\"\"\"Training Summary:

Total Epochs: {len(history['train_loss'])}
Best Epoch: {best_epoch + 1}
Best Val CER: {best_val_cer:.4f}

Model: HTR (plan.md)
Parameters: {total_params:,}
IMG_HEIGHT: {Config.IMG_HEIGHT}px

Architecture:
- CNN: {Config.CNN_CHANNELS}
- Transformer: d={Config.D_MODEL}
  heads={Config.N_HEADS}, layers={Config.N_LAYERS}
- Decoder: CTC
\"\"\"
ax7.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
        verticalalignment='center')

plt.savefig(Config.CHARTS_DIR / 'training_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: training_overview.png")"""))

# Cell 14: Test evaluation
nb.cells.append(nbf.v4.new_code_cell("""# Test set evaluation
model.load_state_dict(torch.load(Config.MODEL_DIR / 'best_model.pth')['model_state_dict'])
model.eval()

test_preds = []
test_targets = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        images = batch['images'].to(Config.DEVICE)
        logits = model(images)
        preds = ctc_decode_greedy(logits, idx_to_char)
        test_preds.extend(preds)
        test_targets.extend(batch['label_texts'])

test_cer = calculate_cer(test_preds, test_targets)
test_wer = calculate_wer(test_preds, test_targets)

print(f"\\nTest Results:")
print(f"  CER: {test_cer:.4f}")
print(f"  WER: {test_wer:.4f}")

for i, (pred, target) in enumerate(zip(test_preds, test_targets)):
    print(f"\\nTest Sample {i+1}:")
    print(f"  Target ({len(target)} chars): '{target[:100]}...'")
    print(f"  Pred   ({len(pred)} chars): '{pred[:100]}...'")
    print(f"  CER: {calculate_cer([pred], [target]):.4f}")"""))

# Cell 15: Sample predictions visualization
nb.cells.append(nbf.v4.new_code_cell("""# Sample predictions visualization
fig, axes = plt.subplots(2, 2, figsize=(18, 10))
axes = axes.flatten()

sample_batch = next(iter(val_loader))
with torch.no_grad():
    images = sample_batch['images'].to(Config.DEVICE)
    logits = model(images)
    predictions = ctc_decode_greedy(logits, idx_to_char)

for idx in range(min(4, len(sample_batch['images']))):
    img = sample_batch['images'][idx].squeeze().cpu().numpy()
    target = sample_batch['label_texts'][idx]
    pred = predictions[idx]

    axes[idx].imshow(img, cmap='gray')
    axes[idx].axis('off')

    cer = calculate_cer([pred], [target])

    title = f"Sample {idx+1} (CER: {cer:.3f})\\n"
    title += f"Target: '{target[:60]}...'\\n"
    title += f"Pred:   '{pred[:60]}...'"
    axes[idx].set_title(title, fontsize=10, loc='left')

plt.tight_layout()
plt.savefig(Config.CHARTS_DIR / 'sample_predictions.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: sample_predictions.png")"""))

# Cell 16: Save artifacts
nb.cells.append(nbf.v4.new_code_cell("""# Save all artifacts
history_df = pd.DataFrame(history)
history_df.to_csv(Config.MODEL_DIR / 'training_history.csv', index=False)

metrics = {
    'best_epoch': int(best_epoch + 1),
    'best_val_cer': float(best_val_cer),
    'test_cer': float(test_cer),
    'test_wer': float(test_wer),
    'total_params': int(total_params),
    'model': 'HTR (from plan.md)',
    'img_height': Config.IMG_HEIGHT,
    'd_model': Config.D_MODEL,
    'n_layers': Config.N_LAYERS
}

with open(Config.MODEL_DIR / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

config_dict = {
    'model': 'HTR',
    'img_height': Config.IMG_HEIGHT,
    'cnn_channels': Config.CNN_CHANNELS,
    'd_model': Config.D_MODEL,
    'n_heads': Config.N_HEADS,
    'n_layers': Config.N_LAYERS,
    'vocab_size': vocab_size,
    'batch_size': Config.BATCH_SIZE,
    'learning_rate': Config.LEARNING_RATE,
    'epochs': Config.EPOCHS
}

with open(Config.MODEL_DIR / 'config.json', 'w') as f:
    json.dump(config_dict, f, indent=2)

print("\\n✓ All artifacts saved!")
print(f"  Best model: {Config.MODEL_DIR / 'best_model.pth'}")
print(f"  Vocabulary: {Config.MODEL_DIR / 'vocabulary.json'}")
print(f"  Training history: {Config.MODEL_DIR / 'training_history.csv'}")
print(f"  Metrics: {Config.MODEL_DIR / 'metrics.json'}")
print(f"  Config: {Config.MODEL_DIR / 'config.json'}")
print(f"  Charts: {Config.CHARTS_DIR}")
print(f"  Preprocessing viz: {Config.CHARTS_DIR / 'preprocessing'}")"""))

# Cell 17: Final summary
nb.cells.append(nbf.v4.new_markdown_cell("""## Training Complete!

### Check These Visualizations:
1. **Preprocessing stages**: `charts/preprocessing/` - Shows transformation at each step
2. **Training overview**: `charts/training_overview.png` - Loss, CER, metrics
3. **Sample predictions**: `charts/sample_predictions.png` - Visual examples

### Next Steps:
If CER is still high (~98%), this is expected with only 9 training documents attempting full-page recognition. To improve:
- Collect 100+ more labeled documents
- Implement line-level segmentation
- Add beam search decoder with language model
- Use transfer learning from pre-trained models

The preprocessing visualization will show if images are being transformed correctly!"""))

# Save notebook
with open('/Users/ismatsamadov/azeri_handwriting_detection/notebooks/training.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("✓ Created clean training notebook with preprocessing visualization")
print(f"  Total cells: {len(nb.cells)}")
print(f"  Markdown cells: {sum(1 for c in nb.cells if c.cell_type == 'markdown')}")
print(f"  Code cells: {sum(1 for c in nb.cells if c.cell_type == 'code')}")
