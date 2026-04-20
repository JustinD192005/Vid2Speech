# vid2voc — Speech Reconstruction from Silent Lip Video

A deep learning system that reconstructs **intelligible, speaker-preserving speech** directly from silent lip-movement video — no text transcription, no lip-reading model, no pretrained backbone.

Built and trained entirely on a **free-tier Google Colab T4 GPU**.



---

## What It Does

Given a silent `.mpg` video of someone speaking, vid2voc:

1. Extracts mouth-region frames from the video
2. Encodes them into a 128-d latent vector using a 3D-CNN
3. Decodes that vector into parametric speech features (spectral envelope, pitch, aperiodicity, voicing)
4. Synthesises a waveform using the WORLD vocoder

The output audio **sounds like the original speaker** — vocal identity is preserved through the source-filter decomposition.

---

## Architecture

| Component | Details |
|-----------|---------|
| **Video Encoder** | 5 × Conv3D (64→128→256→512→128), BN, ReLU, Dropout3D, AdaptiveAvgPool → 128-d |
| **SP Decoder** | MLP 128→256→512→480 → 60×8 log-MFSC spectral envelope |
| **AP Decoder** | MLP 128→128→64→40 → 5×8 aperiodicity (VUV-gated) |
| **F0 Decoder** | Linear 128→8, sigmoid (VUV-gated) |
| **VUV Decoder** | Linear 128→8, sigmoid → voiced/unvoiced mask |
| **LSP Decoder** | MLP 128→64→17 → auxiliary (training-only regulariser) |
| **Loss** | Weighted MSE: λ_SP=600, λ_AP=50, λ_F0=10, λ_VUV=10, λ_LSP=1 |


---

## Results

Trained for 100 epochs on 5 speakers (s1–s5) from the GRID corpus.

- **Training loss:** 0.090 → 0.008
- **Validation loss:** 0.040 → 0.010
- **No overfitting** — train and val curves stay parallel

### Sample Reconstruction (GT vs Predicted)

<img width="1589" height="1180" alt="image" src="https://github.com/user-attachments/assets/80e3e937-877f-4790-845b-12f066de490f" />


**Key findings:**
- Speaker identity preserved in every sample
- Voicing structure (silence vs speech) correctly recovered
- Formant positions match ground truth
- Audio sounds robotic (MSE temporal smearing + WORLD excitation), leading to over-smoothed signals where the predicted waveform does not closely match the original waveform
- F0 peaks underestimated by ~20–25%
- Silence regions partially filled due to VUV mask leakage

---

## Dataset

This project uses the [GRID Audiovisual Corpus](https://spandh.dcs.shef.ac.uk/gridcorpus/).

**You must download it yourself** — the dataset is not included in this repository.

1. Go to https://spandh.dcs.shef.ac.uk/gridcorpus/
2. Download video and audio files for speakers s1–s5
3. Organise them as:
```
Speech Reconstruction Dataset/
├── s1/
│   ├── video/    # .mpg files
│   └── audio/    # .wav files
├── s2/
│   ├── video/
│   └── audio/
...
└── s5/
    ├── video/
    └── audio/
```
4. Zip the folder and upload to Google Drive as `Speech Reconstruction Dataset.zip`

---

## How to Run

### 1. Open in Colab

Upload `vid2voc.ipynb` to Google Colab with a **T4 GPU runtime**.

### 2. Install dependencies

```bash
pip install -q librosa soundfile opencv-python-headless pysptk tqdm pyworld
```

### 3. Mount Drive & set paths

Update the paths in Cell 3 to point to your dataset zip and checkpoint directory:

```python
ZIP_PATH = "/content/drive/MyDrive/Speech Reconstruction Dataset.zip"
CHECKPOINT_DIR = "/content/drive/MyDrive/vid2voc_checkpoints"
```

### 4. Run cells in order

- **Cells 1–4:** Setup, data loading, pair matching
- **Cells 5–8:** Configuration, feature extraction, normalisation
- **Cells 9–11:** Dataset, caching, model definition
- **Cells 12–13:** Loss function, training loop
- **Cells 14–16:** Training history, reconstruction, inference

Training takes ~2–3 hours for 100 epochs with cached features.

---

## Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `NUM_FRAMES` | 75 | Video frames per clip |
| `FRAME_H × FRAME_W` | 64 × 96 | Mouth crop resolution |
| `SAMPLE_RATE` | 50,000 Hz | Audio sampling rate |
| `SP_DIM` | 60 | Mel-frequency spectral coefficients |
| `AP_DIM` | 5 | Aperiodicity bands |
| `LSP_DIM` | 17 | Line spectral pairs |
| `BATCH_SIZE` | 16 | Training batch size |
| `LEARNING_RATE` | 0.0001 | Adam learning rate |
| `DROPOUT_P` | 0.2 | Dropout probability |

---

## Project Structure

```
vid2voc/
├── vid2voc.ipynb          # Full pipeline (Colab notebook)
├── requirements.txt       # Python dependencies
├── .gitignore             # Excludes checkpoints, data, cache
├── LICENSE                # MIT License
├── images/
│   ├── model_architecture.png
│   ├── system_pipeline.png
│   └── sample_comparison.png
└── README.md              # This file
```

---

## References

1. M. Cooke et al., "An audio-visual corpus for speech perception and automatic speech recognition," *JASA*, 2006.
2. M. Morise et al., "WORLD: A vocoder-based high-quality speech synthesis system," *IEICE Trans.*, 2016.
3. A. Ephrat and S. Peleg, "Vid2speech: Speech reconstruction from silent video," *ICASSP*, 2017.
4. K. R. Prajwal et al., "Learning individual speaking styles for accurate lip to speech synthesis," *CVPR*, 2020.
5. M. Kim et al., "Lip-to-speech synthesis in the wild with multi-task learning," *ICASSP*, 2023.
6. N. Sahipjohn et al., "RobustL2S: Speaker-specific lip-to-speech synthesis exploiting self-supervised representations," *APSIPA ASC*, 2023.
7. Y. Yemini et al., "LipVoicer: Generating speech from silent videos guided by lip reading," *ICLR*, 2024.

---

## Acknowledgements

- [GRID Corpus](https://spandh.dcs.shef.ac.uk/gridcorpus/) by Cooke et al.
- [WORLD Vocoder](https://github.com/mmorise/World) by Masanori Morise
- [PyWorld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) Python wrapper
- [PySPTK](https://github.com/r9y9/pysptk) for LSP extraction
- Trained on Google Colab free-tier T4 GPU
