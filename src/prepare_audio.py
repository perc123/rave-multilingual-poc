
#!/usr/bin/env python3
"""
Prepare multilingual audio for RAVE training.
Handles: FLAC, WAV, WEBM formats
Output: 16kHz mono concatenated WAV file
"""

import os
import subprocess
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

def convert_webm_to_wav(webm_file, wav_file):
    """Convert WEBM to WAV using ffmpeg"""
    cmd = [
        'ffmpeg',
        '-i', str(webm_file),
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        '-y',  # overwrite
        str(wav_file)
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=30)
        return True
    except Exception as e:
        print(f"⚠️  WEBM conversion failed: {e}")
        return False

def load_and_normalize(audio_file, target_sr=16000):
    """Load audio file and normalize to 16kHz mono"""
    try:
        y, sr = librosa.load(str(audio_file), sr=target_sr, mono=True)
        return y
    except Exception as e:
        print(f"⚠️  Error loading {audio_file.name}: {e}")
        return None

def main():
    # Paths
    raw_dir = Path("./data/raw")
    processed_dir = Path("./data/processed")
    processed_dir.mkdir(exist_ok=True)
    
    TARGET_DURATION = 1650  # seconds (27.5 minutes)
    SAMPLE_RATE = 16000
    
    print("\n" + "="*60)
    print("🎵 RAVE Audio Preparation Pipeline")
    print("="*60)
    
    # Step 1: Find all audio files
    print("\n📂 Scanning for audio files...")
    audio_files = {
        'flac': list(raw_dir.rglob('*.flac')),
        'wav': list(raw_dir.rglob('*.wav')),
        'webm': list(raw_dir.rglob('*.webm')),
    }
    
    print(f"  ✓ FLAC files: {len(audio_files['flac'])}")
    print(f"  ✓ WAV files: {len(audio_files['wav'])}")
    print(f"  ✓ WEBM files: {len(audio_files['webm'])}")
    
    # Step 2: Convert WEBM to WAV
    if audio_files['webm']:
        print(f"\n🔄 Converting {len(audio_files['webm'])} WEBM files to WAV...")
        webm_dir = processed_dir / "webm_converted"
        webm_dir.mkdir(exist_ok=True)
        
        for webm_file in tqdm(audio_files['webm']):
            wav_file = webm_dir / f"{webm_file.stem}.wav"
            if not wav_file.exists():
                convert_webm_to_wav(webm_file, wav_file)
        
        # Add converted files to wav list
        audio_files['wav'].extend(webm_dir.glob('*.wav'))
    
    # Combine all audio files
    all_files = audio_files['flac'] + audio_files['wav']
    print(f"\n📊 Total audio files: {len(all_files)}")
    
    # Step 3: Select files for target duration
    print(f"\n⏱️  Selecting files for ~{TARGET_DURATION/60:.1f} minutes...")
    random.shuffle(all_files)
    
    selected_files = []
    cumulative_duration = 0
    
    for audio_file in tqdm(all_files, desc="Scanning durations"):
        try:
            duration = librosa.get_duration(filename=str(audio_file))
            if cumulative_duration + duration <= TARGET_DURATION * 1.2:
                selected_files.append(audio_file)
                cumulative_duration += duration
                if cumulative_duration >= TARGET_DURATION:
                    break
        except:
            continue
    
    print(f"✅ Selected {len(selected_files)} files")
    print(f"📈 Total duration: {cumulative_duration/60:.1f} minutes")
    
    # Step 4: Normalize and concatenate
    print("\n🔄 Normalizing to 16kHz mono...")
    all_audio = []
    
    for audio_file in tqdm(selected_files, desc="Processing"):
        audio = load_and_normalize(audio_file, SAMPLE_RATE)
        if audio is not None:
            all_audio.append(audio)
    
    if not all_audio:
        print("❌ No audio files processed!")
        return
    
    # Concatenate
    print("\n🔗 Concatenating audio...")
    training_audio = np.concatenate(all_audio)
    
    # Normalize levels
    max_val = np.max(np.abs(training_audio))
    if max_val > 0:
        training_audio = training_audio / max_val
    
    # Save
    output_path = processed_dir / "training_data.wav"
    sf.write(str(output_path), training_audio, SAMPLE_RATE)
    
    print(f"\n✅ Saved: {output_path}")
    print(f"   📁 File size: {output_path.stat().st_size / (1024**2):.1f} MB")
    print(f"   ⏱️  Duration: {len(training_audio) / SAMPLE_RATE / 60:.1f} minutes")
    print(f"   🎚️  Sample rate: {SAMPLE_RATE} Hz")
    print("\n✨ Ready for RAVE training!\n")

if __name__ == "__main__":
    main()
