# Project Workspace Structure

## Git Repo (this directory)
Contains all source code, configs, and documentation.
~/Documents/GitHub/rave-multilingual-poc/

├── src/              # Python scripts

├── config/           # Training config

├── notebooks/        # Jupyter notebooks

└── README.md

## Working Directory (External)
Large training results and data live here (NOT in git).
~/Documents/Rave_ML/rave_migration_package/

├── rave_data/

│   └── audio/training_data.wav         (27.5 min source)

├── rave_project/

│   ├── runs/                           (Training checkpoints)

│   │   └── multilingual_voice_e18d54798e/

│   │       └── version_2/checkpoints/

│   │           ├── multilingual_voice_e18d54798e.ts  ⭐ (Model)

│   │           └── epoch-epoch=24989.ckpt

│   ├── original_test.wav               (Validation audio)

│   └── reconstruction_test.wav

└── output/                              (Generated samples)

├── rave_generated_multilingual.wav

└── rave_generated_multilingual.mp3

## Quick Links

**To generate output:**
```bash
cd ~/Documents/Rave_ML
python3 generate_output.py
```

**To submit results:**
```bash
cd ~/Documents/GitHub/rave-multilingual-poc
# Copy outputs here, commit to git
cp ~/Documents/Rave_ML/output/* ./output/
git add output/
git commit -m "Add generated audio output"
```

## File References

- Model: `~/Documents/Rave_ML/rave_migration_package/rave_project/runs/multilingual_voice_e18d54798e/version_2/checkpoints/multilingual_voice_e18d54798e.ts`
- Generated audio: `~/Documents/Rave_ML/output/rave_generated_multilingual.{wav,mp3}`
- Training data: `~/Documents/Rave_ML/rave_migration_package/rave_data/audio/training_data.wav`