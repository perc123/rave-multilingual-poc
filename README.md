
# RAVE Multilingual Proof-of-Concept

Training RAVE (Recurrent Variational Autoencoder) on multilingual voice data.

## Languages
- English (FLAC)
- Kurdish (FLAC)
- Greek (WAV)
- German (WEBM - needs conversion)

## Project Structure
├── data/

│   ├── raw/          # Original downloaded files

│   └── processed/    # Normalized 16kHz mono audio

├── src/              # Python scripts

├── notebooks/        # Jupyter notebooks

├── checkpoints/      # Model checkpoints

├── output/           # Generated audio

└── logs/            # Training logs

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Pipeline
1. Normalize audio (16kHz mono)
2. Train RAVE on GPU (Colab)
3. Generate output
4. Post-process & validate
