"""
=================================================================
CUSTOM LORA IMPLEMENTATION FOR QWEN3-TTS
=================================================================

Since Qwen3-TTS doesn't have official LoRA support, we implement
a custom LoRA wrapper that can be applied to any Linear layer.

This is based on the original LoRA paper:
"LoRA: Low-Rank Adaptation of Large Language Models"
https://arxiv.org/abs/2106.09685
=================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import math


class LoRALayer(nn.Module):
    """
    Custom LoRA layer that wraps around a Linear layer
    
    Args:
        original_layer: The original nn.Linear layer to wrap
        r: Rank of LoRA matrices (lower = fewer parameters)
        lora_alpha: Scaling factor
        lora_dropout: Dropout probability
    """
    def __init__(
        self,
        original_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1
    ):
        super().__init__()
        
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # Store original layer (frozen)
        self.original_layer = original_layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # LoRA matrices
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # A matrix: (in_features, r)
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        # B matrix: (r, out_features)
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        
        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: original_output + LoRA_output
        """
        # Original layer output (frozen)
        original_output = self.original_layer(x)
        
        # LoRA path: x @ A @ B
        lora_output = self.lora_dropout(x) @ self.lora_A @ self.lora_B
        lora_output = lora_output * self.scaling
        
        return original_output + lora_output
    
    def merge_weights(self):
        """Merge LoRA weights into original layer for inference"""
        if self.r > 0:
            # Compute LoRA weight: A @ B * scaling
            lora_weight = (self.lora_A @ self.lora_B) * self.scaling
            
            # Add to original weight
            self.original_layer.weight.data += lora_weight.T
            
            # Zero out LoRA matrices
            self.lora_A.data.zero_()
            self.lora_B.data.zero_()


def inject_lora_layers(
    model: nn.Module,
    target_modules: List[str],
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    verbose: bool = True
) -> nn.Module:
    """
    Inject LoRA layers into the model
    
    Args:
        model: The model to inject LoRA into
        target_modules: List of module names to target (e.g., ["attn.q_proj", "attn.v_proj"])
        r: LoRA rank
        lora_alpha: Scaling factor
        lora_dropout: Dropout rate
        verbose: Print injection details
    
    Returns:
        Model with LoRA layers injected
    """
    
    injection_count = 0
    
    def _inject_recursive(module, prefix=""):
        nonlocal injection_count
        
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Check if this module should be replaced
            should_replace = False
            for target in target_modules:
                if target in full_name and isinstance(child, nn.Linear):
                    should_replace = True
                    break
            
            if should_replace:
                # Replace with LoRA layer
                lora_layer = LoRALayer(
                    original_layer=child,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout
                )
                setattr(module, name, lora_layer)
                injection_count += 1
                
                if verbose:
                    print(f"✅ Injected LoRA: {full_name} (in={child.in_features}, out={child.out_features}, r={r})")
            else:
                # Recurse
                _inject_recursive(child, full_name)
    
    _inject_recursive(model)
    
    if verbose:
        print(f"\n📊 Total LoRA layers injected: {injection_count}")
    
    return model


def count_trainable_parameters(model: nn.Module) -> tuple:
    """
    Count trainable vs total parameters
    
    Returns:
        (trainable_params, total_params, percentage)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    percentage = 100 * trainable / total if total > 0 else 0
    
    return trainable, total, percentage


def print_trainable_parameters(model: nn.Module):
    """Print trainable parameter statistics"""
    trainable, total, percentage = count_trainable_parameters(model)
    
    print("="*60)
    print("PARAMETER STATISTICS")
    print("="*60)
    print(f"Trainable params: {trainable:,}")
    print(f"Total params: {total:,}")
    print(f"Trainable %: {percentage:.4f}%")
    print(f"Memory (trainable): {trainable * 4 / 1024**2:.2f} MB (FP32)")
    print("="*60)


def save_lora_weights(model: nn.Module, save_path: str):
    """
    Save only LoRA weights (not the full model)
    
    Args:
        model: Model with LoRA layers
        save_path: Path to save LoRA weights
    """
    lora_state_dict = {}
    
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            lora_state_dict[f"{name}.lora_A"] = module.lora_A.data
            lora_state_dict[f"{name}.lora_B"] = module.lora_B.data
    
    torch.save(lora_state_dict, save_path)
    print(f"✅ Saved LoRA weights: {save_path}")
    print(f"   Size: {len(lora_state_dict)} tensors")


def load_lora_weights(model: nn.Module, load_path: str):
    """
    Load LoRA weights into model
    
    Args:
        model: Model with LoRA layers (must be already injected)
        load_path: Path to LoRA weights
    """
    lora_state_dict = torch.load(load_path)
    
    loaded_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"
            
            if a_key in lora_state_dict and b_key in lora_state_dict:
                module.lora_A.data = lora_state_dict[a_key]
                module.lora_B.data = lora_state_dict[b_key]
                loaded_count += 1
    
    print(f"✅ Loaded LoRA weights: {load_path}")
    print(f"   Restored {loaded_count} LoRA layers")


# =================================================================
# EXAMPLE USAGE
# =================================================================
if __name__ == "__main__":
    # Example: Create a simple model and inject LoRA
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(512, 512)
            self.layer2 = nn.Linear(512, 256)
            self.layer3 = nn.Linear(256, 128)
        
        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            x = self.layer3(x)
            return x
    
    # Create model
    model = SimpleModel()
    print("Original model:")
    print_trainable_parameters(model)
    
    # Inject LoRA (target only layer1 and layer2)
    target_modules = ["layer1", "layer2"]
    model = inject_lora_layers(
        model,
        target_modules=target_modules,
        r=8,
        lora_alpha=16,
        verbose=True
    )
    
    print("\nAfter LoRA injection:")
    print_trainable_parameters(model)
    
    # Test forward pass
    x = torch.randn(2, 512)
    output = model(x)
    print(f"\nTest forward pass: input {x.shape} → output {output.shape}")
    
    # Save and load LoRA weights
    save_lora_weights(model, "test_lora.pt")
    load_lora_weights(model, "test_lora.pt")
    
    print("\n✅ LoRA implementation test complete!")
