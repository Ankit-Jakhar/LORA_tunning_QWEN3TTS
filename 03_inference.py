"""
=================================================================
INFERENCE SCRIPT FOR FINE-TUNED QWEN3-TTS
=================================================================

Test the fine-tuned model with Indian accent

Usage:
1. Load base Qwen3-TTS model
2. Load LoRA weights
3. Generate speech with Indian accent
=================================================================
"""

import torch
from qwen_tts import Qwen3TTSModel
from lora_implementation import inject_lora_layers, load_lora_weights
from pathlib import Path
import json
from pydub import AudioSegment
import numpy as np


def load_finetuned_model(
    checkpoint_path: str,
    device: str = "cpu"
):
    """
    Load fine-tuned model with LoRA weights
    
    Args:
        checkpoint_path: Path to checkpoint directory
        device: Device to load model on
    
    Returns:
        model: Loaded model with LoRA weights
    """
    checkpoint_dir = Path(checkpoint_path)
    
    print(f"📦 Loading model from: {checkpoint_dir}")
    
    # Load base model
    print("1️⃣  Loading base Qwen3-TTS...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        dtype=torch.float32,
        attn_implementation="eager"
    )
    
    if hasattr(model, 'model'):
        actual_model = model.model
    else:
        actual_model = model
    
    # Load training state to get config
    state_path = checkpoint_dir / "training_state.pt"
    if state_path.exists():
        print("2️⃣  Loading training state...")
        state = torch.load(state_path, map_location=device)
        config = state['config']
        
        # Inject LoRA with same config
        print("3️⃣  Injecting LoRA layers...")
        actual_model = inject_lora_layers(
            actual_model,
            target_modules=config.target_modules,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=0.0,  # No dropout for inference
            verbose=False
        )
        
        # Load LoRA weights
        print("4️⃣  Loading LoRA weights...")
        lora_path = checkpoint_dir / "lora_weights.pt"
        load_lora_weights(actual_model, str(lora_path))
        
        print(f"✅ Model loaded successfully!")
        print(f"   Global step: {state['global_step']}")
        print(f"   Best val loss: {state['best_val_loss']:.4f}")
    else:
        print("⚠️  No training state found, loading LoRA weights only...")
        lora_path = checkpoint_dir / "lora_weights.pt"
        if lora_path.exists():
            load_lora_weights(actual_model, str(lora_path))
        else:
            print("❌ No LoRA weights found!")
            return None
    
    # Move to device
    actual_model.to(device)
    actual_model.eval()
    
    # Put back into wrapper if needed
    if hasattr(model, 'model'):
        model.model = actual_model
        return model
    else:
        return actual_model


def generate_speech(
    model,
    text: str,
    speaker: str = "aiden",
    instruction: str = "A natural Indian English voice with clear pronunciation and warm tone",
    output_path: str = "output.wav"
):
    """
    Generate speech using fine-tuned model
    
    Args:
        model: Fine-tuned Qwen3TTS model
        text: Text to synthesize
        speaker: Speaker ID
        instruction: Voice instruction (detailed tone)
        output_path: Where to save audio
    """
    print(f"\n🎙️  Generating speech...")
    print(f"   Text: {text[:100]}...")
    print(f"   Speaker: {speaker}")
    print(f"   Instruction: {instruction[:80]}...")
    
    try:
        with torch.no_grad():
            result = model.generate_custom_voice(
                text=text,
                speaker=speaker,
                instruction=instruction,
                language="english"
            )
        
        if isinstance(result, tuple) and len(result) == 2:
            wav, sr = result
        else:
            print(f"❌ Unexpected output format: {type(result)}")
            return None
        
        # Convert to numpy
        if torch.is_tensor(wav):
            arr = wav.cpu().numpy()
        else:
            arr = wav
        
        # Flatten if needed
        if len(arr.shape) > 1:
            arr = arr[0] if arr.shape[0] < arr.shape[-1] else arr[:, 0]
        
        # Convert to PCM16
        if arr.dtype != np.int16:
            if arr.dtype in [np.float32, np.float64]:
                max_val = np.abs(arr).max()
                if max_val > 0:
                    arr = arr / max_val
                arr = (arr * 32767).astype(np.int16)
        
        # Save using pydub
        audio = AudioSegment(
            arr.tobytes(),
            frame_rate=int(sr),
            sample_width=2,
            channels=1
        )
        
        audio.export(output_path, format="wav")
        
        duration = len(audio) / 1000.0
        print(f"✅ Audio generated: {output_path}")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Sample rate: {sr} Hz")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# =================================================================
# TEST CASES
# =================================================================
def run_test_cases(model, output_dir="test_outputs"):
    """Run various test cases to evaluate Indian accent"""
    
    Path(output_dir).mkdir(exist_ok=True)
    
    test_cases = [
        {
            "text": "Hello, good morning! Actually, I was thinking to call you from yesterday only regarding the project deadline. Can you please check once and revert?",
            "instruction": "Natural Indian English male voice with warm and professional tone, medium pace, clear pronunciation with typical Indian expressions",
            "filename": "test_1_office.wav"
        },
        {
            "text": "See actually, the issue is that system is not responding properly since morning only. Kindly do the needful and resolve this at the earliest.",
            "instruction": "Slightly frustrated but polite Indian English female voice, medium-fast pace, concerned but maintaining professionalism",
            "filename": "test_2_complaint.wav"
        },
        {
            "text": "One small doubt I am having. This feature is working fine or not? Please confirm once so that we can move forward.",
            "instruction": "Hesitant and questioning Indian English male voice, slow deliberate pace, soft and cooperative tone",
            "filename": "test_3_question.wav"
        },
        {
            "text": "Don't worry at all. We will manage everything. Just give us little bit time only, then it will be done properly.",
            "instruction": "Reassuring and confident Indian English female voice, calm pace, warm and supportive voice quality",
            "filename": "test_4_reassurance.wav"
        }
    ]
    
    print("\n" + "="*60)
    print("RUNNING TEST CASES")
    print("="*60)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}/{len(test_cases)}")
        output_path = Path(output_dir) / test['filename']
        
        generate_speech(
            model=model,
            text=test['text'],
            instruction=test['instruction'],
            output_path=str(output_path)
        )
    
    print("\n" + "="*60)
    print(f"✅ All test cases complete! Check: {output_dir}")
    print("="*60)


# =================================================================
# MAIN
# =================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test fine-tuned Qwen3-TTS")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/content/drive/MyDrive/qwen_tts_checkpoints/best_model",
        help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to synthesize (if not running test cases)"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Natural Indian English voice with warm tone and clear pronunciation",
        help="Voice instruction"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output audio file path"
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run predefined test cases"
    )
    
    args = parser.parse_args()
    
    # Load model
    model = load_finetuned_model(args.checkpoint)
    
    if model is None:
        print("❌ Failed to load model")
        exit(1)
    
    # Run tests or generate single
    if args.run_tests:
        run_test_cases(model)
    elif args.text:
        generate_speech(
            model=model,
            text=args.text,
            instruction=args.instruction,
            output_path=args.output
        )
    else:
        print("⚠️  No text provided and --run-tests not specified")
        print("Usage examples:")
        print('  python 03_inference.py --run-tests')
        print('  python 03_inference.py --text "Your text here" --output output.wav')
