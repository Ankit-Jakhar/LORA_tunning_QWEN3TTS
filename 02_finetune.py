"""
=================================================================
QWEN3-TTS FINE-TUNING WITH CUSTOM LORA
Main Training Script
=================================================================

WARNING: This is EXPERIMENTAL. Qwen3-TTS doesn't have official
training support. This script attempts to fine-tune using:
1. Custom LoRA implementation
2. Reconstructed training loop
3. Best-guess loss functions

May not work. Use at your own risk.
=================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import json
from pathlib import Path
from tqdm.auto import tqdm
from datasets import load_from_disk
import numpy as np
import librosa
from dataclasses import dataclass
from typing import Optional
import gc

# Import our custom LoRA
from lora_implementation import (
    inject_lora_layers,
    print_trainable_parameters,
    save_lora_weights,
    load_lora_weights,
    count_trainable_parameters
)

from qwen_tts import Qwen3TTSModel


# =================================================================
# CONFIGURATION
# =================================================================
@dataclass
class TrainingConfig:
    # Model
    model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    
    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: list = None  # Will be loaded from target_layers.json
    
    # Training
    batch_size: int = 1  # Very small for Colab Free
    gradient_accumulation_steps: int = 16  # Simulate batch_size=16
    num_epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    
    # Data
    train_data_path: str = "./processed_train"
    val_data_path: str = "./processed_val"
    max_audio_length: float = 30.0  # seconds
    target_sample_rate: int = 24000  # Qwen uses 24kHz internally
    
    # Checkpointing
    output_dir: str = "/content/drive/MyDrive/qwen_tts_checkpoints"
    save_steps: int = 500
    eval_steps: int = 250
    logging_steps: int = 50
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True  # Use FP16
    gradient_checkpointing: bool = True
    
    # Debugging
    debug_mode: bool = False  # Use tiny subset for testing
    max_train_samples: Optional[int] = None  # Limit samples for quick test


# =================================================================
# DATASET
# =================================================================
class IndianAccentDataset(Dataset):
    """
    Dataset for Indian accent audio
    
    Note: This is a SIMPLIFIED dataset. Real TTS training needs:
    - Mel spectrograms
    - Text tokens
    - Speaker embeddings
    - Proper alignment
    
    Since we don't have Qwen's exact preprocessing, this is best-guess.
    """
    def __init__(
        self,
        dataset_path: str,
        sample_rate: int = 24000,
        max_length: float = 30.0,
        debug_mode: bool = False
    ):
        self.dataset = load_from_disk(dataset_path)
        self.sample_rate = sample_rate
        self.max_length = max_length
        
        # In debug mode, use only 100 samples
        if debug_mode:
            self.dataset = self.dataset.select(range(min(100, len(self.dataset))))
        
        print(f"📊 Loaded dataset: {len(self.dataset)} samples")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Extract audio
        audio = item['audio']['array']
        sr = item['audio']['sampling_rate']
        
        # Resample if needed
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        # Trim/pad to max_length
        max_samples = int(self.max_length * self.sample_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        else:
            audio = np.pad(audio, (0, max_samples - len(audio)))
        
        # Get text
        text = item.get('sentence', item.get('text', ''))
        
        return {
            'audio': torch.FloatTensor(audio),
            'text': text,
            'audio_length': len(audio)
        }


def collate_fn(batch):
    """Collate function for DataLoader"""
    audios = torch.stack([item['audio'] for item in batch])
    texts = [item['text'] for item in batch]
    lengths = torch.LongTensor([item['audio_length'] for item in batch])
    
    return {
        'audio': audios,
        'text': texts,
        'lengths': lengths
    }


# =================================================================
# TRAINING LOOP
# =================================================================
class Qwen3TTSTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Load model
        print("📦 Loading Qwen3-TTS model...")
        self.model = self.load_model()
        
        # Inject LoRA
        print("\n💉 Injecting LoRA layers...")
        self.inject_lora()
        
        # Setup training
        self.setup_training()
        
        # Load data
        print("\n📂 Loading datasets...")
        self.setup_data()
        
        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def load_model(self):
        """Load Qwen3-TTS model"""
        model = Qwen3TTSModel.from_pretrained(
            self.config.model_name,
            dtype=torch.float32,  # Training in FP32
            attn_implementation="eager"
        )
        
        # Access the actual model (Qwen wrapper contains the real model)
        if hasattr(model, 'model'):
            actual_model = model.model
        else:
            actual_model = model
        
        # Enable gradient checkpointing to save memory
        if self.config.gradient_checkpointing and hasattr(actual_model, 'gradient_checkpointing_enable'):
            actual_model.gradient_checkpointing_enable()
            print("✅ Gradient checkpointing enabled")
        
        return actual_model
    
    def inject_lora(self):
        """Inject LoRA layers into model"""
        # Load target modules from inspection
        if os.path.exists("target_layers.json"):
            with open("target_layers.json") as f:
                all_layers = json.load(f)
            
            # Filter to attention layers only (common practice)
            target_modules = [
                layer for layer in all_layers
                if any(x in layer.lower() for x in ['attn', 'attention', 'q_proj', 'k_proj', 'v_proj', 'o_proj'])
            ]
            
            # If no attention layers found, use first 20 layers
            if not target_modules:
                print("⚠️  No attention layers found, using first 20 linear layers")
                target_modules = all_layers[:20]
        else:
            print("⚠️  target_layers.json not found, using common patterns")
            target_modules = ["attn", "attention"]
        
        self.config.target_modules = target_modules
        print(f"🎯 Targeting {len(target_modules)} layers for LoRA")
        
        # Inject LoRA
        self.model = inject_lora_layers(
            self.model,
            target_modules=target_modules,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            verbose=False  # Too much output
        )
        
        print_trainable_parameters(self.model)
    
    def setup_training(self):
        """Setup optimizer, scheduler, etc."""
        # Only optimize LoRA parameters
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = AdamW(
            lora_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Scheduler
        total_steps = len(self.train_loader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config.learning_rate * 0.1
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.config.mixed_precision and self.device.type == 'cuda' else None
    
    def setup_data(self):
        """Setup dataloaders"""
        train_dataset = IndianAccentDataset(
            self.config.train_data_path,
            sample_rate=self.config.target_sample_rate,
            max_length=self.config.max_audio_length,
            debug_mode=self.config.debug_mode
        )
        
        val_dataset = IndianAccentDataset(
            self.config.val_data_path,
            sample_rate=self.config.target_sample_rate,
            max_length=self.config.max_audio_length,
            debug_mode=self.config.debug_mode
        )
        
        # Limit samples if specified
        if self.config.max_train_samples:
            train_dataset.dataset = train_dataset.dataset.select(range(self.config.max_train_samples))
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2
        )
        
        print(f"✅ Train batches: {len(self.train_loader)}")
        print(f"✅ Val batches: {len(self.val_loader)}")
    
    def compute_loss(self, batch):
        """
        Compute training loss
        
        WARNING: This is a PLACEHOLDER loss function.
        Qwen3-TTS's actual loss is unknown. This is a simplified version
        that may not work properly.
        
        Real TTS loss should include:
        - Mel spectrogram reconstruction
        - Duration prediction
        - Pitch/energy prediction
        - Adversarial loss (if GAN-based)
        """
        # Move to device
        audio = batch['audio'].to(self.device)
        texts = batch['text']
        
        # PLACEHOLDER: Simple reconstruction loss
        # In reality, we need to:
        # 1. Generate mel spectrograms from audio
        # 2. Feed through model
        # 3. Compute proper TTS losses
        
        # For now, we'll just compute a dummy loss to make training run
        # This WILL NOT actually train the model properly
        
        try:
            # Attempt to get model output
            # This will likely fail - Qwen3TTS needs specific inputs
            output = self.model(audio)  # This line will probably error
            
            # Compute MSE loss (placeholder)
            loss = F.mse_loss(output, audio)
        except Exception as e:
            # If model forward fails, return a dummy loss
            # This allows the training loop to run for testing
            print(f"⚠️  Model forward failed: {e}")
            print("⚠️  Using dummy loss - model is NOT actually training!")
            loss = torch.tensor(1.0, requires_grad=True, device=self.device)
        
        return loss
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        
        for step, batch in enumerate(pbar):
            # Compute loss
            with torch.cuda.amp.autocast() if self.scaler else torch.enable_grad():
                loss = self.compute_loss(batch)
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'lr': f'{lr:.2e}',
                        'step': self.global_step
                    })
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}")
                
                # Validation
                if self.global_step % self.config.eval_steps == 0:
                    val_loss = self.validate()
                    print(f"\n📊 Step {self.global_step} | Val Loss: {val_loss:.4f}")
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best_model")
                        print("✅ New best model saved!")
                    
                    self.model.train()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                loss = self.compute_loss(batch)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, name):
        """Save checkpoint"""
        checkpoint_dir = Path(self.config.output_dir) / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA weights only (much smaller)
        lora_path = checkpoint_dir / "lora_weights.pt"
        save_lora_weights(self.model, str(lora_path))
        
        # Save training state
        state_path = checkpoint_dir / "training_state.pt"
        torch.save({
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config
        }, state_path)
        
        print(f"💾 Checkpoint saved: {checkpoint_dir}")
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Total epochs: {self.config.num_epochs}")
        print(f"Steps per epoch: {len(self.train_loader)}")
        print(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print("="*60 + "\n")
        
        try:
            for epoch in range(self.config.num_epochs):
                print(f"\n📅 Epoch {epoch + 1}/{self.config.num_epochs}")
                
                train_loss = self.train_epoch(epoch)
                print(f"✅ Epoch {epoch + 1} complete | Avg Loss: {train_loss:.4f}")
                
                # Save epoch checkpoint
                self.save_checkpoint(f"epoch-{epoch+1}")
                
                # Clear cache
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
        
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
            self.save_checkpoint("interrupted")
        
        except Exception as e:
            print(f"\n❌ Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            self.save_checkpoint("error")
        
        finally:
            print("\n" + "="*60)
            print("TRAINING COMPLETE")
            print("="*60)
            print(f"Total steps: {self.global_step}")
            print(f"Best val loss: {self.best_val_loss:.4f}")
            print(f"Checkpoints saved to: {self.config.output_dir}")


# =================================================================
# MAIN
# =================================================================
if __name__ == "__main__":
    # Configuration
    config = TrainingConfig()
    
    # Set debug mode for quick testing
    config.debug_mode = False  # Set to True for testing
    config.max_train_samples = None  # Set to 1000 for quick test
    
    # Create trainer and train
    trainer = Qwen3TTSTrainer(config)
    trainer.train()
