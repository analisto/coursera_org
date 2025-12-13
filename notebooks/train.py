#!/usr/bin/env python3
"""
HTR-Lite Training Script for Azeri Handwriting Recognition
Complete standalone training pipeline with CTC loss and Transformer architecture
"""

import os
import json
import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import pandas as pd
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Training configuration and hyperparameters"""

    # Paths
    BASE_DIR = Path("/Users/ismatsamadov/azeri_handwriting_detection")
    DATA_DIR = BASE_DIR / "data"
    IMAGES_DIR = DATA_DIR / "images"
    LABELS_DIR = DATA_DIR / "labels"
    OUTPUT_DIR = BASE_DIR / "model_artifacts"
    CHARTS_DIR = BASE_DIR / "charts"

    # Image preprocessing
    IMG_HEIGHT = 256
    IMG_WIDTH = None  # Variable width

    # Training hyperparameters
    BATCH_SIZE = 2
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Model architecture
    EMBEDDING_DIM = 256
    NUM_HEADS = 8
    NUM_ENCODER_LAYERS = 4
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1

    # Data split ratios
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # Special tokens
    PAD_TOKEN = '<PAD>'
    BLANK_TOKEN = '<BLANK>'

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Random seed
    SEED = 42

    def __init__(self):
        """Initialize configuration and create directories"""
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.CHARTS_DIR.mkdir(parents=True, exist_ok=True)

        # Set random seeds
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.SEED)

    def to_dict(self):
        """Convert config to dictionary for saving"""
        return {
            'img_height': self.IMG_HEIGHT,
            'batch_size': self.BATCH_SIZE,
            'num_epochs': self.NUM_EPOCHS,
            'learning_rate': self.LEARNING_RATE,
            'weight_decay': self.WEIGHT_DECAY,
            'embedding_dim': self.EMBEDDING_DIM,
            'num_heads': self.NUM_HEADS,
            'num_encoder_layers': self.NUM_ENCODER_LAYERS,
            'dim_feedforward': self.DIM_FEEDFORWARD,
            'dropout': self.DROPOUT,
            'train_ratio': self.TRAIN_RATIO,
            'val_ratio': self.VAL_RATIO,
            'test_ratio': self.TEST_RATIO,
            'seed': self.SEED,
            'device': str(self.DEVICE)
        }


# ============================================================================
# Image Preprocessing
# ============================================================================

class ImagePreprocessor:
    """Handles image loading and preprocessing with single log message"""

    _preprocessing_logged = False  # Class variable for single log

    def __init__(self, target_height: int = 256):
        self.target_height = target_height

    @staticmethod
    def convert_heic_to_rgb(image_path: str) -> np.ndarray:
        """Convert HEIC/HEIF images to RGB numpy array"""
        try:
            from PIL import Image
            import pillow_heif

            # Register HEIF opener with PIL
            pillow_heif.register_heif_opener()

            # Open and convert
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return np.array(img)
        except Exception as e:
            raise RuntimeError(f"Failed to load HEIC image {image_path}: {str(e)}")

    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from various formats"""
        if not ImagePreprocessor._preprocessing_logged:
            print("Preprocessing images (HEIC conversion, resizing, normalization)...")
            ImagePreprocessor._preprocessing_logged = True

        if image_path.lower().endswith(('.heic', '.heif')):
            img = self.convert_heic_to_rgb(image_path)
        else:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def preprocess(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess image:
        1. Load image (handle HEIC)
        2. Convert to grayscale
        3. Resize maintaining aspect ratio
        4. Normalize to [0, 1]
        """
        # Load image
        img = self.load_image(image_path)

        # Convert to grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Resize maintaining aspect ratio
        h, w = img.shape
        aspect_ratio = w / h
        new_width = int(self.target_height * aspect_ratio)
        img = cv2.resize(img, (new_width, self.target_height))

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Add channel dimension: (H, W) -> (1, H, W)
        img = np.expand_dims(img, axis=0)

        return img


# ============================================================================
# Vocabulary Builder
# ============================================================================

class VocabularyBuilder:
    """Build character vocabulary from label files"""

    def __init__(self, labels_dir: Path, pad_token: str = '<PAD>', blank_token: str = '<BLANK>'):
        self.labels_dir = labels_dir
        self.pad_token = pad_token
        self.blank_token = blank_token

    def build(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build vocabulary from all label files"""
        print("\nBuilding vocabulary from label files...")

        # Collect all characters
        all_chars = set()
        label_files = sorted(self.labels_dir.glob("*.txt"))

        for label_file in label_files:
            with open(label_file, 'r', encoding='utf-8') as f:
                text = f.read()
                all_chars.update(text)

        # Create sorted list (excluding newlines)
        chars = sorted([c for c in all_chars if c != '\n'])

        # Build mappings with special tokens
        char2idx = {self.blank_token: 0}  # CTC blank token at index 0
        char2idx[self.pad_token] = 1      # Padding token at index 1

        # Add regular characters
        for idx, char in enumerate(chars, start=2):
            char2idx[char] = idx

        # Create reverse mapping
        idx2char = {idx: char for char, idx in char2idx.items()}

        print(f"Vocabulary size: {len(char2idx)} characters")
        print(f"  - Special tokens: {self.blank_token}, {self.pad_token}")
        print(f"  - Regular characters: {len(chars)}")

        return char2idx, idx2char


# ============================================================================
# Dataset
# ============================================================================

class HTRDataset(Dataset):
    """Dataset for Handwritten Text Recognition"""

    def __init__(
        self,
        image_paths: List[str],
        label_paths: List[str],
        char2idx: Dict[str, int],
        preprocessor: ImagePreprocessor
    ):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.char2idx = char2idx
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.image_paths)

    def encode_text(self, text: str) -> List[int]:
        """Encode text to indices"""
        # Remove newlines
        text = text.replace('\n', '')
        # Encode characters
        encoded = [self.char2idx.get(c, self.char2idx['<PAD>']) for c in text]
        return encoded

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        # Load and preprocess image
        image = self.preprocessor.preprocess(self.image_paths[idx])
        image = torch.FloatTensor(image)

        # Load and encode label
        with open(self.label_paths[idx], 'r', encoding='utf-8') as f:
            text = f.read()

        label = self.encode_text(text)
        label = torch.LongTensor(label)

        # Return image, label, and original image width for CTC
        return image, label, image.shape[2]  # width is at dimension 2


def collate_fn(batch):
    """
    Custom collate function for variable-width images
    Returns:
        - images: padded to max width in batch (B, 1, H, W)
        - labels: padded labels (B, max_label_len)
        - label_lengths: original label lengths (B,)
        - input_lengths: sequence lengths after CNN (B,)
    """
    images, labels, widths = zip(*batch)

    # Find max width and max label length
    max_width = max(widths)
    max_label_len = max(len(l) for l in labels)

    # Pad images to max width
    batch_images = []
    for img in images:
        if img.shape[2] < max_width:
            # Pad on the right
            pad_width = max_width - img.shape[2]
            img = F.pad(img, (0, pad_width, 0, 0), value=0)
        batch_images.append(img)

    batch_images = torch.stack(batch_images)

    # Pad labels
    batch_labels = torch.zeros(len(labels), max_label_len, dtype=torch.long)
    label_lengths = []
    for i, label in enumerate(labels):
        batch_labels[i, :len(label)] = label
        label_lengths.append(len(label))

    label_lengths = torch.LongTensor(label_lengths)

    # Calculate input lengths (after CNN downsampling)
    # CNN reduces width by factor of 4 (2x2 pooling twice)
    input_lengths = torch.LongTensor([w // 4 for w in widths])

    return batch_images, batch_labels, label_lengths, input_lengths


# ============================================================================
# HTR-Lite Model
# ============================================================================

class HTRLiteModel(nn.Module):
    """
    HTR-Lite: CNN + Transformer Encoder + CTC
    Approximately 3.4M parameters
    """

    def __init__(
        self,
        num_classes: int,
        img_height: int = 256,
        embedding_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        super(HTRLiteModel, self).__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # CNN Feature Extractor
        # Input: (B, 1, 256, W) -> Output: (B, 256, 64, W/4)
        self.cnn = nn.Sequential(
            # Block 1: 1 -> 32
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> (B, 32, 128, W/2)

            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> (B, 64, 64, W/4)

            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Calculate feature dimension after CNN
        # Height: 256 -> 128 -> 64 = 64
        # Each position has 256 channels
        self.feature_height = 64
        self.cnn_output_dim = 256 * self.feature_height

        # Linear projection to embedding dimension
        self.projection = nn.Linear(self.cnn_output_dim, embedding_dim)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # CTC Output layer
        self.fc = nn.Linear(embedding_dim, num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input images (B, 1, H, W)
        Returns:
            Log probabilities for CTC (T, B, num_classes)
        """
        # CNN feature extraction
        # (B, 1, H, W) -> (B, 256, H/4, W/4)
        features = self.cnn(x)

        # Reshape: (B, C, H', W') -> (B, W', C*H')
        B, C, H, W = features.shape
        features = features.permute(0, 3, 1, 2)  # (B, W', C, H')
        features = features.reshape(B, W, -1)     # (B, W', C*H')

        # Project to embedding dimension
        features = self.projection(features)  # (B, W', embedding_dim)
        features = self.dropout(features)

        # Add positional encoding
        features = self.positional_encoding(features)

        # Transformer encoding
        encoded = self.transformer(features)  # (B, W', embedding_dim)

        # CTC output
        logits = self.fc(encoded)  # (B, W', num_classes)

        # Permute for CTC: (B, W', num_classes) -> (W', B, num_classes)
        logits = logits.permute(1, 0, 2)

        # Log softmax for CTC
        log_probs = F.log_softmax(logits, dim=2)

        return log_probs


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (B, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# Training Functions
# ============================================================================

class Trainer:
    """Handles model training and evaluation"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler],
        config: Config,
        idx2char: Dict[int, str]
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.idx2char = idx2char

        # CTC Loss
        self.criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_cer': [],
            'val_cer': [],
            'learning_rate': []
        }

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def decode_predictions(self, log_probs: torch.Tensor, input_lengths: torch.Tensor) -> List[str]:
        """Decode CTC predictions to text"""
        predictions = []

        # Get most likely characters at each timestep
        _, pred_indices = log_probs.max(dim=2)  # (T, B)
        pred_indices = pred_indices.permute(1, 0)  # (B, T)

        for i, length in enumerate(input_lengths):
            # Get prediction for this sample
            pred = pred_indices[i, :length].cpu().numpy()

            # CTC collapse: merge repeated characters and remove blanks
            decoded = []
            prev_char = None
            for char_idx in pred:
                if char_idx == 0:  # blank token
                    prev_char = None
                elif char_idx != prev_char:
                    decoded.append(self.idx2char.get(char_idx, ''))
                    prev_char = char_idx

            predictions.append(''.join(decoded))

        return predictions

    def calculate_cer(self, predictions: List[str], targets: List[str]) -> float:
        """Calculate Character Error Rate"""
        total_chars = 0
        total_errors = 0

        for pred, target in zip(predictions, targets):
            # Simple character-level distance
            target = target.replace('\n', '')

            # Levenshtein distance
            errors = self.levenshtein_distance(pred, target)
            total_errors += errors
            total_chars += len(target)

        return total_errors / max(total_chars, 1)

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return Trainer.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} [Train]")

        for images, labels, label_lengths, input_lengths in pbar:
            # Move to device
            images = images.to(self.config.DEVICE)
            labels = labels.to(self.config.DEVICE)
            label_lengths = label_lengths.to(self.config.DEVICE)
            input_lengths = input_lengths.to(self.config.DEVICE)

            # Forward pass
            log_probs = self.model(images)  # (T, B, num_classes)

            # Calculate CTC loss
            loss = self.criterion(log_probs, labels, input_lengths, label_lengths)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Track loss
            total_loss += loss.item()

            # Decode predictions for CER calculation
            predictions = self.decode_predictions(log_probs, input_lengths)

            # Get target texts
            targets = []
            for i in range(len(labels)):
                target_indices = labels[i, :label_lengths[i]].cpu().numpy()
                target_text = ''.join([self.idx2char.get(idx, '') for idx in target_indices])
                targets.append(target_text)

            all_predictions.extend(predictions)
            all_targets.extend(targets)

            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.train_loader)
        cer = self.calculate_cer(all_predictions, all_targets)

        return avg_loss, cer

    def validate(self, epoch: int) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} [Val]  ")

            for images, labels, label_lengths, input_lengths in pbar:
                # Move to device
                images = images.to(self.config.DEVICE)
                labels = labels.to(self.config.DEVICE)
                label_lengths = label_lengths.to(self.config.DEVICE)
                input_lengths = input_lengths.to(self.config.DEVICE)

                # Forward pass
                log_probs = self.model(images)

                # Calculate loss
                loss = self.criterion(log_probs, labels, input_lengths, label_lengths)
                total_loss += loss.item()

                # Decode predictions
                predictions = self.decode_predictions(log_probs, input_lengths)

                # Get targets
                targets = []
                for i in range(len(labels)):
                    target_indices = labels[i, :label_lengths[i]].cpu().numpy()
                    target_text = ''.join([self.idx2char.get(idx, '') for idx in target_indices])
                    targets.append(target_text)

                all_predictions.extend(predictions)
                all_targets.extend(targets)

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.val_loader)
        cer = self.calculate_cer(all_predictions, all_targets)

        return avg_loss, cer

    def train(self):
        """Main training loop"""
        print("\n" + "="*80)
        print("Starting Training")
        print("="*80)
        print(f"Device: {self.config.DEVICE}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"Epochs: {self.config.NUM_EPOCHS}")
        print("="*80 + "\n")

        for epoch in range(self.config.NUM_EPOCHS):
            # Train
            train_loss, train_cer = self.train_epoch(epoch)

            # Validate
            val_loss, val_cer = self.validate(epoch)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_cer'].append(train_cer)
            self.history['val_cer'].append(val_cer)
            self.history['learning_rate'].append(current_lr)

            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train CER: {train_cer:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val CER:   {val_cer:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_model('best_model.pth')
                print(f"  >>> New best model saved! (Val Loss: {val_loss:.4f})")

            print()

        print("="*80)
        print(f"Training completed! Best model from epoch {self.best_epoch+1}")
        print("="*80 + "\n")

    def save_model(self, filename: str):
        """Save model checkpoint"""
        filepath = self.config.OUTPUT_DIR / filename
        torch.save({
            'epoch': self.best_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
        }, filepath)

    def test(self) -> Dict:
        """Evaluate on test set"""
        print("\n" + "="*80)
        print("Evaluating on Test Set")
        print("="*80)

        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        sample_results = []

        with torch.no_grad():
            for batch_idx, (images, labels, label_lengths, input_lengths) in enumerate(tqdm(self.test_loader, desc="Testing")):
                # Move to device
                images = images.to(self.config.DEVICE)
                labels = labels.to(self.config.DEVICE)
                label_lengths = label_lengths.to(self.config.DEVICE)
                input_lengths = input_lengths.to(self.config.DEVICE)

                # Forward pass
                log_probs = self.model(images)

                # Calculate loss
                loss = self.criterion(log_probs, labels, input_lengths, label_lengths)
                total_loss += loss.item()

                # Decode predictions
                predictions = self.decode_predictions(log_probs, input_lengths)

                # Get targets
                targets = []
                for i in range(len(labels)):
                    target_indices = labels[i, :label_lengths[i]].cpu().numpy()
                    target_text = ''.join([self.idx2char.get(idx, '') for idx in target_indices])
                    targets.append(target_text)

                all_predictions.extend(predictions)
                all_targets.extend(targets)

                # Save some samples for visualization
                if batch_idx < 5:
                    for i in range(min(2, len(predictions))):
                        sample_results.append({
                            'image': images[i].cpu().numpy(),
                            'prediction': predictions[i],
                            'target': targets[i]
                        })

        avg_loss = total_loss / len(self.test_loader)
        cer = self.calculate_cer(all_predictions, all_targets)

        print(f"\nTest Results:")
        print(f"  Test Loss: {avg_loss:.4f}")
        print(f"  Test CER:  {cer:.4f}")
        print("="*80 + "\n")

        return {
            'test_loss': avg_loss,
            'test_cer': cer,
            'predictions': all_predictions,
            'targets': all_targets,
            'samples': sample_results
        }


# ============================================================================
# Visualization
# ============================================================================

class Visualizer:
    """Create training visualizations"""

    def __init__(self, config: Config):
        self.config = config

    def plot_training_overview(self, history: Dict):
        """Plot training and validation metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Overview', fontsize=16, fontweight='bold')

        epochs = range(1, len(history['train_loss']) + 1)

        # Loss plot
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # CER plot
        axes[0, 1].plot(epochs, history['train_cer'], 'b-', label='Train CER', linewidth=2)
        axes[0, 1].plot(epochs, history['val_cer'], 'r-', label='Val CER', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Character Error Rate')
        axes[0, 1].set_title('Character Error Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Learning rate plot
        axes[1, 0].plot(epochs, history['learning_rate'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')

        # Best epoch marker
        best_epoch = np.argmin(history['val_loss']) + 1
        axes[1, 1].text(0.5, 0.6, f"Best Epoch: {best_epoch}",
                       ha='center', va='center', fontsize=20, fontweight='bold')
        axes[1, 1].text(0.5, 0.4, f"Best Val Loss: {min(history['val_loss']):.4f}",
                       ha='center', va='center', fontsize=16)
        axes[1, 1].text(0.5, 0.2, f"Best Val CER: {history['val_cer'][best_epoch-1]:.4f}",
                       ha='center', va='center', fontsize=16)
        axes[1, 1].axis('off')

        plt.tight_layout()
        save_path = self.config.CHARTS_DIR / 'training_overview.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved training overview: {save_path}")

    def plot_error_analysis(self, test_results: Dict):
        """Plot error analysis"""
        predictions = test_results['predictions']
        targets = test_results['targets']

        # Calculate per-sample CER
        sample_cers = []
        for pred, target in zip(predictions, targets):
            target = target.replace('\n', '')
            if len(target) > 0:
                cer = Trainer.levenshtein_distance(pred, target) / len(target)
                sample_cers.append(cer)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Error Analysis', fontsize=16, fontweight='bold')

        # CER distribution
        axes[0, 0].hist(sample_cers, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Character Error Rate')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('CER Distribution')
        axes[0, 0].axvline(np.mean(sample_cers), color='red', linestyle='--',
                          linewidth=2, label=f'Mean: {np.mean(sample_cers):.3f}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Error rate by text length
        text_lengths = [len(t.replace('\n', '')) for t in targets]
        axes[0, 1].scatter(text_lengths, sample_cers, alpha=0.5, color='steelblue')
        axes[0, 1].set_xlabel('Text Length (characters)')
        axes[0, 1].set_ylabel('Character Error Rate')
        axes[0, 1].set_title('CER vs Text Length')
        axes[0, 1].grid(True, alpha=0.3)

        # Perfect predictions percentage
        perfect_preds = sum(1 for cer in sample_cers if cer == 0)
        total_preds = len(sample_cers)

        categories = ['Perfect\nPredictions', 'With Errors']
        values = [perfect_preds, total_preds - perfect_preds]
        colors = ['#2ecc71', '#e74c3c']

        axes[1, 0].bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Prediction Accuracy')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        for i, v in enumerate(values):
            axes[1, 0].text(i, v + 0.5, f'{v}\n({v/total_preds*100:.1f}%)',
                           ha='center', va='bottom', fontweight='bold')

        # Statistics summary
        stats_text = f"""
        Total Samples: {total_preds}

        Mean CER: {np.mean(sample_cers):.4f}
        Median CER: {np.median(sample_cers):.4f}
        Std CER: {np.std(sample_cers):.4f}

        Min CER: {np.min(sample_cers):.4f}
        Max CER: {np.max(sample_cers):.4f}

        Perfect Predictions: {perfect_preds} ({perfect_preds/total_preds*100:.1f}%)
        """

        axes[1, 1].text(0.1, 0.5, stats_text,
                       ha='left', va='center', fontsize=12,
                       fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')

        plt.tight_layout()
        save_path = self.config.CHARTS_DIR / 'error_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved error analysis: {save_path}")

    def plot_sample_predictions(self, samples: List[Dict]):
        """Plot sample predictions"""
        n_samples = min(6, len(samples))
        fig, axes = plt.subplots(n_samples, 1, figsize=(15, 3*n_samples))

        if n_samples == 1:
            axes = [axes]

        fig.suptitle('Sample Predictions', fontsize=16, fontweight='bold')

        for i, sample in enumerate(samples[:n_samples]):
            # Display image
            img = sample['image'].squeeze()  # Remove channel dimension
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')

            # Add prediction and target text
            pred_text = sample['prediction']
            target_text = sample['target'].replace('\n', ' ')

            # Calculate CER for this sample
            cer = Trainer.levenshtein_distance(pred_text, target_text) / max(len(target_text), 1)

            title = f"Target:     {target_text}\n"
            title += f"Prediction: {pred_text}\n"
            title += f"CER: {cer:.3f}"

            axes[i].set_title(title, fontsize=10, loc='left',
                            fontfamily='monospace', pad=10)

        plt.tight_layout()
        save_path = self.config.CHARTS_DIR / 'sample_predictions.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved sample predictions: {save_path}")


# ============================================================================
# Data Preparation
# ============================================================================

def prepare_data(config: Config) -> Tuple[List, List, List, List, List, List]:
    """Prepare train/val/test splits"""
    print("\nPreparing dataset...")

    # Get all label files
    label_files = sorted(config.LABELS_DIR.glob("*.txt"))
    print(f"Found {len(label_files)} label files")

    # Create corresponding image paths
    image_paths = []
    label_paths = []

    for label_file in label_files:
        # Get base name (e.g., az_formal_letter_01)
        base_name = label_file.stem

        # Find corresponding images (01.HEIC, 02.HEIC, etc.)
        # Count lines in label file to know how many images
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        num_images = len(lines)

        for i in range(1, num_images + 1):
            img_name = f"{i:02d}.HEIC"
            img_path = config.IMAGES_DIR / img_name

            if img_path.exists():
                image_paths.append(str(img_path))
                label_paths.append(str(label_file))

    print(f"Found {len(image_paths)} image-label pairs")

    # Shuffle data
    indices = list(range(len(image_paths)))
    random.shuffle(indices)

    image_paths = [image_paths[i] for i in indices]
    label_paths = [label_paths[i] for i in indices]

    # Split data
    n_total = len(image_paths)
    n_train = int(n_total * config.TRAIN_RATIO)
    n_val = int(n_total * config.VAL_RATIO)

    train_images = image_paths[:n_train]
    train_labels = label_paths[:n_train]

    val_images = image_paths[n_train:n_train+n_val]
    val_labels = label_paths[n_train:n_train+n_val]

    test_images = image_paths[n_train+n_val:]
    test_labels = label_paths[n_train+n_val:]

    print(f"Split: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")

    return train_images, train_labels, val_images, val_labels, test_images, test_labels


# ============================================================================
# Main
# ============================================================================

def main():
    """Main training pipeline"""
    print("\n" + "="*80)
    print("HTR-Lite Training Pipeline")
    print("Azeri Handwriting Recognition")
    print("="*80)

    # Initialize config
    config = Config()

    # Build vocabulary
    vocab_builder = VocabularyBuilder(config.LABELS_DIR, config.PAD_TOKEN, config.BLANK_TOKEN)
    char2idx, idx2char = vocab_builder.build()

    # Save vocabulary
    vocab_path = config.OUTPUT_DIR / 'vocabulary.json'
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump({
            'char2idx': char2idx,
            'idx2char': {str(k): v for k, v in idx2char.items()},
            'vocab_size': len(char2idx)
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved vocabulary to {vocab_path}")

    # Prepare data
    train_images, train_labels, val_images, val_labels, test_images, test_labels = prepare_data(config)

    # Create preprocessor
    preprocessor = ImagePreprocessor(target_height=config.IMG_HEIGHT)

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = HTRDataset(train_images, train_labels, char2idx, preprocessor)
    val_dataset = HTRDataset(val_images, val_labels, char2idx, preprocessor)
    test_dataset = HTRDataset(test_images, test_labels, char2idx, preprocessor)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"Data loaders created (batch_size={config.BATCH_SIZE})")

    # Create model
    print("\nInitializing model...")
    model = HTRLiteModel(
        num_classes=len(char2idx),
        img_height=config.IMG_HEIGHT,
        embedding_dim=config.EMBEDDING_DIM,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_ENCODER_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT
    ).to(config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created: {total_params:,} parameters ({trainable_params:,} trainable)")

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # Create learning rate scheduler
    total_steps = len(train_loader) * config.NUM_EPOCHS
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos'
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        idx2char=idx2char
    )

    # Train model
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time

    # Test model
    test_results = trainer.test()

    # Save metrics
    metrics = {
        'final_train_loss': trainer.history['train_loss'][-1],
        'final_val_loss': trainer.history['val_loss'][-1],
        'final_train_cer': trainer.history['train_cer'][-1],
        'final_val_cer': trainer.history['val_cer'][-1],
        'best_val_loss': trainer.best_val_loss,
        'best_epoch': trainer.best_epoch + 1,
        'test_loss': test_results['test_loss'],
        'test_cer': test_results['test_cer'],
        'training_time_seconds': training_time,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params
    }

    metrics_path = config.OUTPUT_DIR / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Save config
    config_path = config.OUTPUT_DIR / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"Saved config to {config_path}")

    # Save training history
    history_df = pd.DataFrame({
        'epoch': range(1, config.NUM_EPOCHS + 1),
        'train_loss': trainer.history['train_loss'],
        'val_loss': trainer.history['val_loss'],
        'train_cer': trainer.history['train_cer'],
        'val_cer': trainer.history['val_cer'],
        'learning_rate': trainer.history['learning_rate']
    })

    history_path = config.OUTPUT_DIR / 'training_history.csv'
    history_df.to_csv(history_path, index=False)
    print(f"Saved training history to {history_path}")

    # Create visualizations
    print("\nCreating visualizations...")
    visualizer = Visualizer(config)
    visualizer.plot_training_overview(trainer.history)
    visualizer.plot_error_analysis(test_results)
    visualizer.plot_sample_predictions(test_results['samples'])

    # Print final summary
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Training time: {training_time/60:.2f} minutes")
    print(f"Best validation loss: {trainer.best_val_loss:.4f} (epoch {trainer.best_epoch+1})")
    print(f"Test CER: {test_results['test_cer']:.4f}")
    print(f"\nModel artifacts saved to: {config.OUTPUT_DIR}")
    print(f"Charts saved to: {config.CHARTS_DIR}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
