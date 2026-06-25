#!/usr/bin/env python3
"""
Generate novel audio from trained RAVE model.
Samples random latent codes to create 4-5 minutes of output.
"""

import torch
import soundfile as sf
import numpy as np
from pathlib import Path
import json
import subprocess

def generate_from_rave(model_path, output_duration_seconds=300, num_seeds=3):
    """Generate audio by sampling from model's latent space."""
    
    SAMPLE_RATE = 16000
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    print("\n" + "="*60)
    print("🎵 RAVE Audio Generation")
    print("="*60)

    # Load model
    print(f"\n📦 Loading model: {model_path}")
    model = torch.jit.load(model_path)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"✅ Model loaded on {device}")

    all_audio = []

    for seed in range(num_seeds):
        torch.manual_seed(seed)
        duration_per_seed = output_duration_seconds // num_seeds
        latent_time_steps = int((SAMPLE_RATE * duration_per_seed) / 32)
        latent_dim = 128

        print(f"\n🌱 Seed {seed}: Generating {duration_per_seed}s...")

        with torch.no_grad():
            latent = torch.randn(1, latent_dim, latent_time_steps, device=device)
            audio = model.decode(latent)

        audio_np = audio.cpu().numpy()[0, 0] if audio.dim() == 4 else audio.cpu().numpy()[0]
        print(f"   ✓ {len(audio_np) / SAMPLE_RATE:.1f}s generated")
        all_audio.append(audio_np)

    # Concatenate & normalize
    full_audio = np.concatenate(all_audio)
    max_val = np.max(np.abs(full_audio))
    full_audio = full_audio / max_val * 0.95

    # Save WAV
    output_wav = output_dir / "rave_generated.wav"
    sf.write(str(output_wav), full_audio, SAMPLE_RATE)
    print(f"\n✅ WAV: {output_wav} ({len(full_audio)/SAMPLE_RATE/60:.1f} min)")

    # Convert to MP3
    output_mp3 = output_dir / "rave_generated.mp3"
    subprocess.run(['ffmpeg', '-i', str(output_wav), '-b:a', '192k', '-y', str(output_mp3)], 
                   capture_output=True)
    print(f"✅ MP3: {output_mp3} ({output_mp3.stat().st_size/(1024**2):.1f} MB)")

if __name__ == "__main__":
    model_path = Path("checkpoints/multilingual_voice_e18d54798e.ts")
    generate_from_rave(model_path)