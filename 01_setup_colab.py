"""
=================================================================
QWEN3-TTS FINE-TUNING ON INDIAN ACCENT DATA
Setup Notebook for Google Colab
=================================================================

IMPORTANT: This is EXPERIMENTAL - Qwen3-TTS doesn't have official
fine-tuning support. We're implementing custom LoRA approach.

Steps:
1. Mount Google Drive
2. Install dependencies
3. Download Indian accent dataset
4. Prepare data
5. Model inspection

Author: Experimental LoRA Implementation
Warning: May not work - Qwen3-TTS architecture is proprietary
=================================================================
"""

# =================================================================
# CELL 1: Check GPU and Mount Drive
# =================================================================
# Check GPU
!nvidia-smi

# Mount Google Drive for checkpoints
from google.colab import drive
drive.mount('/content/drive')

# Create checkpoint directory
import os
checkpoint_dir = "/content/drive/MyDrive/qwen_tts_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
print(f"✅ Checkpoints will be saved to: {checkpoint_dir}")


# =================================================================
# CELL 2: Install Dependencies
# =================================================================
# Install required packages
!pip install -q torch torchvision torchaudio
!pip install -q transformers accelerate
!pip install -q datasets
!pip install -q librosa soundfile
!pip install -q pydub
!pip install -q wandb  # Optional: for tracking
!pip install -q bitsandbytes  # For 8-bit training

# Install Qwen TTS
!pip install -q qwen-tts

print("✅ All dependencies installed")


# =================================================================
# CELL 3: Download Indian Accent Dataset
# =================================================================
from datasets import load_dataset
import os

# Download dataset
print("📥 Downloading Indian Accent dataset...")
print("This will take 10-20 minutes for ~95k samples")

dataset = load_dataset("WillHeld/india_accent_cv", split="train")

# Save to disk
dataset_path = "./india_accent_cv_train"
dataset.save_to_disk(dataset_path)

print(f"✅ Dataset downloaded: {len(dataset)} samples")
print(f"💾 Saved to: {dataset_path}")

# Show sample
print("\n📊 Sample data:")
print(dataset[0])


# =================================================================
# CELL 4: Inspect Dataset Structure
# =================================================================
import pandas as pd

# Check what columns we have
print("Dataset columns:", dataset.column_names)
print("\nDataset features:", dataset.features)

# Check a few samples
print("\n📝 First 3 samples:")
for i in range(3):
    sample = dataset[i]
    print(f"\nSample {i+1}:")
    print(f"  Audio path: {sample.get('audio', {}).get('path', 'N/A')}")
    print(f"  Text: {sample.get('sentence', sample.get('text', 'N/A'))[:100]}...")
    if 'audio' in sample and 'array' in sample['audio']:
        print(f"  Audio shape: {sample['audio']['array'].shape}")
        print(f"  Sample rate: {sample['audio']['sampling_rate']}")


# =================================================================
# CELL 5: Filter and Prepare Data
# =================================================================
import numpy as np
from datasets import Dataset

def filter_valid_samples(example):
    """Filter out invalid or too short/long samples"""
    try:
        # Check if audio exists
        if 'audio' not in example or example['audio'] is None:
            return False
        
        # Check audio length (between 1-30 seconds)
        audio_array = example['audio']['array']
        sr = example['audio']['sampling_rate']
        duration = len(audio_array) / sr
        
        if duration < 1.0 or duration > 30.0:
            return False
        
        # Check if text exists
        text = example.get('sentence', example.get('text', ''))
        if not text or len(text) < 5:
            return False
        
        return True
    except:
        return False

print("🔍 Filtering dataset...")
filtered_dataset = dataset.filter(filter_valid_samples)

print(f"✅ Filtered: {len(dataset)} → {len(filtered_dataset)} samples")

# Reduce to manageable size for Colab Free (optional)
# Uncomment if you want to use subset for faster experimentation
# filtered_dataset = filtered_dataset.select(range(10000))  # Use only 10k samples
# print(f"📊 Using subset: {len(filtered_dataset)} samples")

# Split into train/validation
split_dataset = filtered_dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split_dataset['train']
val_dataset = split_dataset['test']

print(f"✅ Train samples: {len(train_dataset)}")
print(f"✅ Val samples: {len(val_dataset)}")


# =================================================================
# CELL 6: Load and Inspect Qwen3-TTS Model
# =================================================================
import torch
from qwen_tts import Qwen3TTSModel

print("📦 Loading Qwen3-TTS model...")
print("⚠️  This will download ~3.4GB model")

# Load model
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    dtype=torch.float32,  # Use float32 for training
    attn_implementation="eager"
)

# Move to CPU first (we'll handle GPU in training script)
device = "cpu"
if hasattr(model, 'model') and model.model is not None:
    model.model.to(device)

print("✅ Model loaded")

# Inspect model architecture
print("\n" + "="*60)
print("MODEL ARCHITECTURE INSPECTION")
print("="*60)

def inspect_model_architecture(model, max_depth=3):
    """Recursively inspect model structure"""
    def _inspect(module, name="", depth=0):
        if depth > max_depth:
            return
        
        indent = "  " * depth
        print(f"{indent}{name}: {module.__class__.__name__}")
        
        # Check for trainable parameters
        if hasattr(module, 'weight'):
            if module.weight.requires_grad:
                print(f"{indent}  → Trainable: {module.weight.shape}")
        
        # Recurse into children
        for child_name, child_module in module.named_children():
            _inspect(child_module, child_name, depth + 1)
    
    if hasattr(model, 'model'):
        _inspect(model.model)
    else:
        _inspect(model)

inspect_model_architecture(model)

# Find potential LoRA target layers
print("\n" + "="*60)
print("POTENTIAL LORA TARGET LAYERS")
print("="*60)

target_layers = []
for name, module in model.named_modules() if hasattr(model, 'model') else model.model.named_modules():
    if isinstance(module, torch.nn.Linear):
        target_layers.append(name)

print(f"Found {len(target_layers)} Linear layers")
print("First 20 layers:")
for i, layer in enumerate(target_layers[:20]):
    print(f"  {i+1}. {layer}")

# Save target layers for training script
import json
with open("target_layers.json", "w") as f:
    json.dump(target_layers, f)

print(f"\n✅ Saved {len(target_layers)} target layers to target_layers.json")


# =================================================================
# CELL 7: Memory Check
# =================================================================
import torch

print("💾 Memory Check:")
print(f"GPU Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Current Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"Current Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

# Calculate model size
def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

if hasattr(model, 'model'):
    size_mb = get_model_size(model.model)
else:
    size_mb = get_model_size(model)

print(f"\nModel Size: {size_mb:.2f} MB ({size_mb/1024:.2f} GB)")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")


# =================================================================
# CELL 8: Save Prepared Data for Training
# =================================================================
# Save processed datasets
train_dataset.save_to_disk("./processed_train")
val_dataset.save_to_disk("./processed_val")

print("✅ Datasets saved:")
print(f"  Train: ./processed_train ({len(train_dataset)} samples)")
print(f"  Val: ./processed_val ({len(val_dataset)} samples)")

print("\n" + "="*60)
print("SETUP COMPLETE!")
print("="*60)
print("\nNext steps:")
print("1. Review the target layers found above")
print("2. Run the training script (finetune.py)")
print("3. Monitor checkpoints in Google Drive")
print("\n⚠️  WARNING: Training may take 8-12+ hours on Colab Free")
print("⚠️  Consider using Colab Pro for faster training")
