# Qwen3-TTS Fine-tuning on Indian Accent Data

## ⚠️ CRITICAL WARNINGS

**THIS IS AN EXPERIMENTAL IMPLEMENTATION**

1. **Qwen3-TTS does NOT have official fine-tuning support**
2. **This code attempts custom LoRA implementation** - it may not work
3. **The training loop uses placeholder loss functions** - actual TTS losses are unknown
4. **Success rate: ~30-40%** - be prepared for failure
5. **Recommended alternative: Use XTTS v2 instead** (has proven fine-tuning support)

**If your manager insists on Qwen3-TTS:**
- Allocate 2-3 weeks for research + implementation
- Budget for Colab Pro ($10/month minimum)
- Have a backup plan ready

---

## 📋 What This Code Does

Attempts to fine-tune Qwen3-TTS-12Hz-1.7B-CustomVoice on Indian accent English audio using:
- Custom LoRA (Low-Rank Adaptation) implementation
- Indian Accent Common Voice dataset (95.6k samples)
- Memory-optimized training for Google Colab Free

---

## 📁 File Structure

```
qwen-tts-finetuning/
├── 01_setup_colab.py          # Colab setup + data download
├── lora_implementation.py      # Custom LoRA layers
├── 02_finetune.py             # Main training script
├── 03_inference.py            # Testing script
└── README.md                  # This file
```

---

## 🚀 Quick Start (Google Colab)

### Step 1: Setup (30 minutes)

```python
# In Colab, create new notebook and run:

# Upload 01_setup_colab.py to Colab
# Then run it cell by cell
%run 01_setup_colab.py
```

**This will:**
- ✅ Mount Google Drive
- ✅ Install dependencies
- ✅ Download 95.6k Indian accent samples (~15-20 min)
- ✅ Inspect Qwen3-TTS architecture
- ✅ Prepare data for training

### Step 2: Upload Training Files

Upload these files to Colab:
- `lora_implementation.py`
- `02_finetune.py`

### Step 3: Start Training

```python
# In Colab
%run 02_finetune.py
```

**Expected:**
- Training time: **8-12 hours** on Colab Free T4
- Checkpoints saved to Google Drive every 500 steps
- Can resume if Colab disconnects

### Step 4: Test Model

```python
# Upload 03_inference.py
%run 03_inference.py --run-tests
```

---

## 🔧 Configuration

Edit `TrainingConfig` in `02_finetune.py`:

```python
@dataclass
class TrainingConfig:
    # LoRA settings
    lora_r: int = 8              # Lower = less parameters (4-16)
    lora_alpha: int = 16         # Scaling (usually 2x r)
    lora_dropout: float = 0.1    # Dropout rate
    
    # Training
    batch_size: int = 1          # Keep at 1 for Colab Free
    gradient_accumulation_steps: int = 16  # Effective batch = 16
    num_epochs: int = 3          # 3-5 epochs recommended
    learning_rate: float = 2e-4  # 1e-4 to 5e-4 range
    
    # Memory optimization
    gradient_checkpointing: bool = True  # Must be True
    mixed_precision: bool = True         # Use FP16
    
    # Quick testing
    debug_mode: bool = False     # Set True for 100 samples only
    max_train_samples: int = None  # Set 1000 for quick test
```

---

## 💾 Checkpoint Management

Checkpoints are saved to Google Drive:
```
/content/drive/MyDrive/qwen_tts_checkpoints/
├── checkpoint-500/
│   ├── lora_weights.pt        # LoRA only (~50MB)
│   └── training_state.pt      # Optimizer state
├── checkpoint-1000/
├── best_model/                # Best validation loss
└── epoch-1/                   # End of each epoch
```

**To resume training:**
```python
# Modify 02_finetune.py to load from checkpoint
# (See resume training section in code)
```

---

## 🧪 Testing Your Model

### Option 1: Run Test Cases
```python
python 03_inference.py --run-tests --checkpoint /path/to/checkpoint
```

Generates 4 test samples with different Indian accent scenarios.

### Option 2: Custom Text
```python
python 03_inference.py \
    --checkpoint /path/to/best_model \
    --text "Hello, kindly do the needful and revert at earliest." \
    --instruction "Professional Indian English voice with polite tone" \
    --output my_audio.wav
```

---

## 📊 Expected Results

### If Everything Works (30% chance):
- ✅ Model generates audio with Indian accent
- ✅ Natural Indian English expressions
- ✅ Better than base Qwen3-TTS for Indian speakers

### Most Likely Outcome (70% chance):
- ⚠️ Model trains but doesn't improve quality
- ⚠️ Loss decreases but output sounds same as base
- ⚠️ LoRA layers train but model architecture incompatible

### Why It Might Fail:
1. **Unknown architecture** - LoRA targeting wrong layers
2. **Wrong loss function** - Placeholder loss doesn't match real TTS objective
3. **Data format mismatch** - Qwen expects specific preprocessing
4. **Missing components** - Duration/pitch predictors not trained

---

## 🐛 Common Issues & Solutions

### Issue 1: CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```python
# Reduce batch size (already at 1)
# Reduce gradient accumulation
gradient_accumulation_steps = 8  # Down from 16

# Use smaller dataset
max_train_samples = 5000  # Instead of full 95k

# Enable more aggressive optimizations
# (Already enabled in code)
```

### Issue 2: Colab Disconnects
Training auto-saves checkpoints. To resume:
```python
# In 02_finetune.py, modify __main__:
trainer = Qwen3TTSTrainer(config)
# trainer.load_checkpoint("checkpoint-500")  # Uncomment this
trainer.train()
```

### Issue 3: Model Doesn't Improve
This is expected - try:
1. Increase LoRA rank: `lora_r = 16`
2. Target more layers
3. Train for more epochs: `num_epochs = 5`

**Or:** Switch to XTTS v2 (proven to work)

---

## 💡 Alternative: XTTS v2 (Recommended)

If Qwen3-TTS fine-tuning fails, use XTTS v2:

**Advantages:**
- ✅ Official fine-tuning support
- ✅ Proven to work on Colab Free
- ✅ Documented training scripts
- ✅ Active community
- ✅ Similar quality to Qwen3-TTS

**Setup:**
```bash
pip install TTS
# Use official Coqui-TTS fine-tuning scripts
```

---

## 📞 Getting Help

1. **Check model forward pass works:**
   ```python
   # Test if base model generates audio
   from qwen_tts import Qwen3TTSModel
   model = Qwen3TTSModel.from_pretrained(...)
   wav, sr = model.generate_custom_voice(...)
   # If this fails, fine-tuning won't work
   ```

2. **Monitor training loss:**
   - Loss should decrease over time
   - If stuck at 1.0, it's the dummy placeholder loss

3. **Compare outputs:**
   - Generate sample from base model
   - Generate same text from fine-tuned model
   - Listen for any difference in accent

---

## 📈 Success Metrics

**Minimum viable success:**
- [ ] Training completes without crashes
- [ ] Loss decreases from initial value
- [ ] Checkpoints save successfully
- [ ] Inference script generates audio
- [ ] Audio has slight Indian accent improvement

**Good success:**
- [ ] Clear Indian accent in all outputs
- [ ] Natural Indian English expressions
- [ ] Better than base model for Indian speakers

**Excellent success (unlikely):**
- [ ] Perfect Indian accent on all test cases
- [ ] Comparable to human Indian speech
- [ ] Production-ready quality

---

## ⏱️ Time Estimates

| Task | Time (Colab Free) | Time (Colab Pro) |
|------|-------------------|------------------|
| Setup + Download | 30 min | 20 min |
| First training attempt | 8-12 hrs | 4-6 hrs |
| Debugging | 2-3 days | 1-2 days |
| Hyperparameter tuning | 3-5 days | 2-3 days |
| **Total** | **1-2 weeks** | **5-7 days** |

---

## 📝 Manager Communication Template

```
Dear [Manager],

I've implemented the Qwen3-TTS fine-tuning pipeline as requested. 
Please note:

CURRENT STATUS:
- ✅ Code implemented with custom LoRA
- ✅ Dataset downloaded (95.6k samples)
- ⚠️  Training started but results uncertain

CHALLENGES:
1. Qwen3-TTS has no official training support
2. Using experimental custom LoRA implementation  
3. Success rate estimated at 30-40%

TIMELINE:
- Initial results: 3-5 days
- Production-ready (if successful): 2-3 weeks
- Backup plan (XTTS v2): 1 week guaranteed

RECOMMENDATION:
Consider running both approaches in parallel:
- Continue Qwen3-TTS experiment (high risk, high reward)
- Start XTTS v2 as backup (low risk, proven results)

BUDGET REQUEST:
- Colab Pro: $10/month (will save 50% time)

I'll update you in 48 hours with initial training results.
```

---

## 🎯 Next Steps

1. **Run setup script** - Download data and inspect model
2. **Start training** - Run for 6-8 hours
3. **Check checkpoint** - Test if model generates audio
4. **Evaluate quality** - Compare with base model
5. **Decide:** Continue or switch to XTTS v2

---

## 📚 References

- Qwen3-TTS: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
- LoRA Paper: https://arxiv.org/abs/2106.09685
- Indian Accent Dataset: https://huggingface.co/datasets/WillHeld/india_accent_cv
- XTTS v2 (Alternative): https://github.com/coqui-ai/TTS

---

## ⚖️ License

This is experimental research code. Use at your own risk.

---

**Good luck! 🍀**

**Remember: If this doesn't work after 1 week, switch to XTTS v2.**
