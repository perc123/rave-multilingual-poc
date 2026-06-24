# RAVE Multilingual Proof-of-Concept

Training [IRCAM's RAVE](https://github.com/acids-ircam/rave) (Realtime Audio Variational autoEncoder) on multilingual voice data, trained locally on an Ubuntu machine with an NVIDIA Quadro RTX 5000. The end product is a `.ts` (TorchScript) model that runs in real time inside Max/MSP (`nn~`), SuperCollider (`nn`), PureData, or any `nn_tilde`-compatible host.

This document covers the actual local setup and pipeline as run, including several non-obvious bugs in the `acids-rave` package that had to be fixed to get training to complete. The original plan (see `notebooks/rave_training_colab.ipynb`) was Colab-based; this repo's code was adapted to run entirely on local hardware instead.

## Result

A complete model trained for 300,000 steps (~16 hours, with one restart) on 27.5 minutes of source audio. Exported and verified working:

```
~/rave_project/runs/multilingual_voice_e18d54798e/version_2/checkpoints/multilingual_voice_e18d54798e.ts
```

## Hardware & environment

- **GPU**: NVIDIA Quadro RTX 5000, 16GB VRAM (driver 595.71.05, CUDA 13.2)
- **Python**: 3.10.20, in a dedicated venv at `~/rave_env` (not conda — conda exists on the box but wasn't used for this)
- **PyTorch**: 2.5.1+cu121
- **RAVE package**: `acids-rave` 2.3.1, installed editable from a clone of the official repo at `~/rave` (`github.com/acids-ircam/rave`, commit `f048ec4`) — installed editable specifically so the source bugs below could be patched in place
- **cached-conv**: 2.5.0, **gin-config**: 0.5.0

### Why a separate `~/rave` checkout instead of plain `pip install acids-rave`

Several bugs (see below) required source patches. An editable install (`pip install -e .`) of a git clone makes those patches take effect immediately without reinstalling.

## Directory layout

This is split across the git repo (small, code/config only) and several local-only directories (large, gitignored or simply outside the repo — audio data and model checkpoints don't belong in git):

| Path | Contents |
|---|---|
| `~/rave-multilingual-poc/` (this repo) | Notebook, prep script, config, this README |
| `~/rave/` | Editable clone of IRCAM's `acids-rave`, with local bugfix patches (not in this repo — it's a vendored dependency) |
| `~/rave_env/` | Python venv for everything RAVE-related |
| `~/rave_data/audio/training_data.wav` | The 16kHz mono source audio actually used for training |
| `~/rave_data/preprocessed/` | LMDB dataset produced by `rave preprocess` (~51MB) |
| `~/rave_project/runs/multilingual_voice_e18d54798e/` | Training run: TensorBoard logs (`version_0/1/2`), `config.gin`, checkpoints, the exported `.ts` |
| `~/rave_project/train.log` | stdout/stderr of the training process |
| `~/rave_project/test_inference.py` | Standalone inference smoke test (see below) |

## Data

The source audio (`~/rave_data/audio/training_data.wav`) is 16kHz mono, ~27.5 minutes, already mixed/concatenated and normalized. `src/prepare_audio.py` in this repo is the script intended to *produce* a file like this from raw multilingual recordings (FLAC/WAV/WEBM, originally English, Kurdish, Greek, German per early project notes) — but the raw source files it expects under `data/raw/` are not present on this machine and were never committed to git (large-binary `.gitignore` rules exist for them, but no raw audio was ever checked in). In practice, `training_data.wav` was already prepared by the time local training started; treat `prepare_audio.py` as documentation of how it *was* (or should be) built, not as a script you can currently re-run end-to-end without first sourcing raw audio into `data/raw/`.

Preprocessing chunks the audio into 131,072-sample (8.192s) windows and stores them in an LMDB at 16kHz:

```bash
source ~/rave_env/bin/activate
rave preprocess \
  --input_path ~/rave_data/audio \
  --output_path ~/rave_data/preprocessed \
  --sampling_rate 16000 \
  --num_signal 131072 \
  --nolazy \
  --dyndb
```

Result: 100 chunks, split 98 train / 2 validation by the RAVE CLI automatically. The 2-example validation set is small enough that the validation loss is a noisy, low-confidence signal — see "Known limitations" below.

**Gotcha**: `rave preprocess` uses absl boolean flags (`--lazy`/`--nolazy`), not `--lazy false`/`--lazy true`. Passing `--lazy false` silently *enables* lazy mode (it parses as bare `--lazy`, with `false` becoming a stray positional argument) and produces a near-empty LMDB containing only a file pointer instead of real decoded/chunked audio. Always use `--nolazy` explicitly.

## Setup

```bash
python3 -m venv ~/rave_env
source ~/rave_env/bin/activate
pip install "setuptools<81"        # see bug #1 below — newer setuptools breaks pytorch_lightning
git clone https://github.com/acids-ircam/rave.git ~/rave
pip install -e ~/rave
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

## Training

```bash
source ~/rave_env/bin/activate
nohup rave train \
  --config v2 \
  --db_path ~/rave_data/preprocessed \
  --name multilingual_voice \
  --channels 1 \
  --max_steps 300000 \
  --val_every 10000 \
  --gpu 0 \
  > ~/rave_project/train.log 2>&1 &
disown
```

Run this from `~/rave_project` so the `runs/` directory it creates lands there. `--channels 1` is required — see bug #3 below.

Throughput observed: ~2.3 steps/sec sustained, GPU at 95-100% utilization, ~8.8GB/16GB VRAM. 300,000 steps took **~16 hours** of actual training time (plus ~1.2 hours lost to a crash and restart — see bug #5).

### Monitoring

Training logs progress bars to `train.log` but not loss values (those go to TensorBoard only):

```bash
tail -f ~/rave_project/train.log
tensorboard --logdir ~/rave_project/runs
```

Or pull the latest scalars directly:

```python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator('~/rave_project/runs/multilingual_voice_e18d54798e/version_2')
ea.Reload()
for tag in ea.Tags()['scalars']:
    print(tag, ea.Scalars(tag)[-1])
```

Checkpoints save every `--val_every` steps to `.../version_N/checkpoints/`, plus a running `best.ckpt` (by validation loss — see limitations, this isn't necessarily the most-trained model).

### Export

```bash
source ~/rave_env/bin/activate
rave export --run ~/rave_project/runs/multilingual_voice_e18d54798e
```

This loads the **most recently modified** checkpoint (not `best.ckpt`) and writes `<run_name>.ts` next to it. No `--cached` flag exists in this version of the CLI (it appears in some Colab-era examples but isn't a real flag here — passing it is a fatal flags-parsing error).

### Testing the exported model

`~/rave_project/test_inference.py` loads the `.ts`, encodes/decodes a 10-second clip from the source audio, and writes both the original and reconstruction as WAV files for listening:

```bash
source ~/rave_env/bin/activate
python3 ~/rave_project/test_inference.py
# -> ~/rave_project/original_test.wav
# -> ~/rave_project/reconstruction_test.wav
```

Final run: input `[1, 1, 160000]` → latent `[1, 8, 79]` (only ~8 of the configured 128 latent dims carry signal at 95% fidelity) → output `[1, 1, 161792]` (decoder adds a few thousand samples of padding — harmless), reconstruction MSE 0.00135 against the original waveform.

## Bugs found in `acids-rave` 2.3.1 and fixes applied

All patches were made directly in the `~/rave` editable checkout. None of this is Colab-specific — the same crashes would happen on any platform with this dataset/config combination.

1. **`rave preprocess --lazy false` doesn't disable lazy mode.** Absl boolean flags require `--nolazy`, not `--lazy false`. Fixed by using `--nolazy`/`--dyndb` with no value.

2. **`setuptools` ≥81 removed `pkg_resources`**, which `pytorch_lightning` (via `lightning_fabric`) imports unconditionally, causing `ModuleNotFoundError: No module named 'pkg_resources'` on the very first `rave train` invocation. Fixed by pinning `pip install "setuptools<81"` in `rave_env`.

3. **`--channels` flag default zeroes the decoder.** `scripts/train.py` calls `rave.dataset.get_training_channels()` to auto-detect the dataset's channel count and binds it to `RAVE.n_channels` via gin — but then separately calls `rave.RAVE(n_channels=FLAGS.channels)` using the raw `--channels` CLI flag (default `0`), which **overrides** the gin binding. In `GeneratorV2.__init__`, `data_size = data_size * n_channels` becomes `16 * 0 = 0`, so the decoder's final conv layer gets `out_channels=0` and crashes immediately with `RuntimeError: cannot reshape tensor of 0 elements...` inside `weight_norm`. Workaround: always pass `--channels 1` explicitly (or the true channel count) — never rely on the `0` default with this version.

4. **Lightning's sanity check crashes before the receptive field is known.** `RAVE.receptive_field` is computed lazily, inside `validation_epoch_end`, *after* the validation batches for that epoch have already run. PyTorch Lightning's sanity check runs a couple of validation batches before training starts, so the very first `validation_step` call always happens with `receptive_field == 0`. First fix attempt: added `num_sanity_val_steps=0` to the `pl.Trainer(...)` call in `scripts/train.py`. This only suppressed the symptom during sanity-check — see bug #5.

5. **`validation_step` never crops `x`/`y` to equal length — at all, ever.** Unlike `training_step` (which crops via `rave.core.valid_signal_crop` once the receptive field is known), `validation_step` in `rave/model.py` passes the raw encoder input and decoder output straight into `audio_distance(x, y)`. Encode→decode does not exactly preserve signal length for this architecture, so the very first **real** validation epoch (not just the sanity check) crashes with `RuntimeError: The size of tensor a (706) must match the size of tensor b (709)...`. Because no checkpoint exists until the first validation epoch succeeds, this crash loses all training progress since the last (or first) checkpoint — in this run, ~1.2 hours of unsaved training. Fix: patched `validation_step` to crop both tensors to `min(x.shape[-1], y.shape[-1])` before computing the distance:
   ```python
   min_len = min(x.shape[-1], y.shape[-1])
   x = x[..., :min_len]
   y = y[..., :min_len]
   distance = self.audio_distance(x, y)
   ```
   After this fix, training ran cleanly through all 30 validation checkpoints to completion.

6. **Exported model declared the wrong sample rate.** `v1.gin` sets `SAMPLING_RATE = 44100` as a default, and nothing in the local pipeline ever overrode it to match the actual 16000 Hz training data. This value only affects STFT loss-frequency weighting during training (cosmetic) — but it also becomes the `.sr` attribute baked into the exported `.ts` file, which real-time hosts (`nn~` etc.) read to decide how to resample incoming audio. A mismatched `sr` would cause the model to receive audio at the wrong effective rate in any DAW, producing pitch/speed artifacts. Fixed by editing `SAMPLING_RATE = 44100` → `16000` directly in the run's saved `config.gin` and re-running `rave export` (confirmed via `torch.jit.load(...).sr == 16000` afterward). This is a metadata-only fix — it doesn't change model weights or behavior in Python, only what gets reported to host software.

## Training results

Final metrics at step 299,999 (epoch 24,999):

| Metric | Start (~step 2,700) | End (step 299,999) |
|---|---|---|
| `multiband_spectral_distance` | 5.17 | 3.37 |
| `fullband_spectral_distance` | 5.30 | 3.56 |
| `validation` | 4.25 (first measurement) | 4.59 |
| `regularization` (KL) | 0.006 | 0.70 |

`beta_factor` (KL warmup weight) ramped from ~0 to its target 0.05 over the first 20,000 steps as configured.

## Known limitations

- **27.5 minutes of source audio is short** for RAVE — expect a model that can faithfully reproduce the timbral character of what it saw, but with limited diversity and likely audible artifacts on material that differs much from the training set.
- **The 2-example validation split is not a reliable quality signal.** Validation loss never improved past its very first measurement throughout training even though the training-set reconstruction loss kept dropping — this is almost certainly an artifact of the tiny validation set's noise, not evidence of overfitting. `best.ckpt` (selected by lowest validation loss) is therefore the *least*-trained usable checkpoint, not the best one; `rave export` defaults to the most recently modified checkpoint instead, which is what you want.
- **Listening is the real test.** MSE and spectral-distance numbers confirm the model isn't broken, but say little about perceptual quality — use `test_inference.py`'s output WAVs (or your own clips) and listen.

## Reproducing / continuing

To retrain from scratch with different settings, repeat the preprocess → train → export steps above with different flags (e.g. a different `--config` such as `v2_small` for faster/lower-quality iteration, or a longer `--max_steps`). To resume a stopped run from its last checkpoint, pass `--ckpt <path-to-checkpoint>` to `rave train`.

To get a better model, the highest-leverage change is more and more varied source audio — re-run (or write) an audio-preparation step that produces a longer `training_data.wav`, then redo `rave preprocess`.
