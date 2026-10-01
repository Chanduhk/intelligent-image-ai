# Intelligent Image Compression & Analysis System

**AI-Based Image Compression and Intelligent Image Analysis using Autoencoders**

---

## Overview

A complete AI pipeline where a single **CNN Autoencoder** serves as the central
feature-extraction layer for image compression, classification, anomaly detection,
similarity search, and continuous learning — all deployed through a Streamlit dashboard.

### Architecture

```
CIFAR-10 → Preprocessing → CNN Encoder → Latent (64×4×4 = 1,024 values)
                                                  │
         ┌────────────────────────────────────────┤
         ↓                    ↓                   ↓
    CNN Decoder         Random Forest      Isolation Forest
    Reconstruction      Classification     Anomaly Detection
         ↓                    ↓                   ↓
  MSE / PSNR / SSIM   Class + Confidence   Normal/Anomalous
         └────────────────────────────────────────┤
                                                  ↓
                                       Specification Engine
                                                  ↓
                                       Explanation Engine
                                                  ↓
                                      Continuous Learning
                                                  ↓
                                  Validation → Deployment Gate
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Run deployment check

```bash
python check.py
```

Expected output: `STATUS: FULLY READY FOR DEPLOYMENT`

### 3. Launch the dashboard

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

> **Current deployed model: v3** — trained for 30 epochs on CIFAR-10.

---

## Training

> **Note:** Training is only required if you want to train a new model version.
> The repository already includes a fully trained v3 model.

### Full training (recommended, ~90 min on CPU)

```bash
python train.py --epochs 30
```

### Training options

```bash
python train.py --epochs 30 --batch-size 128 --data-dir ./data/raw

# Resume interrupted training from last checkpoint:
python train.py --resume --epochs 30

# Quick test run (1 epoch, ~3-4 min):
python train.py --epochs 1
```

> **Windows / CPU note:** Each epoch takes ~150–200 seconds on CPU with
> `batch_size=128`. A full 30-epoch run takes ~90 minutes.
> The system saves a checkpoint after every epoch
> (`models/autoencoder/checkpoints/checkpoint_epoch_NNN.pth`) so training
> can be safely resumed at any point.

### Evaluate a trained model

```bash
python evaluate.py
```

---

## Model Architecture

### CNN Autoencoder

| Layer | Operation | Output Shape |
|-------|-----------|-------------|
| Input | — | 3 × 32 × 32 |
| Encoder Stage 1 | Conv2D(3→32) + BatchNorm + ReLU + MaxPool | 32 × 16 × 16 |
| Encoder Stage 2 | Conv2D(32→64) + BatchNorm + ReLU + MaxPool | 64 × 8 × 8 |
| Encoder Stage 3 | Conv2D(64→64) + BatchNorm + ReLU + MaxPool | **64 × 4 × 4** |
| Decoder Stage 1 | ConvTranspose2D(64→64) + BatchNorm + ReLU | 64 × 8 × 8 |
| Decoder Stage 2 | ConvTranspose2D(64→32) + BatchNorm + ReLU | 32 × 16 × 16 |
| Decoder Stage 3 | ConvTranspose2D(32→3) + Sigmoid | **3 × 32 × 32** |

**Compression:** 3,072 → 1,024 values = **3.0× ratio**

### Downstream Models

| Model | Type | Input | Output |
|-------|------|-------|--------|
| Classifier | Random Forest (300 trees) | Latent vector (1,024) | Class + confidence |
| Anomaly Detector | Isolation Forest (200 trees) | Latent vector (1,024) | Score + Normal/Anomalous |
| Similarity Engine | Cosine similarity | Latent vector (1,024) | Top-k similar images |

---

## Model Artifacts

After successful training the following files are created:

```
models/
├── autoencoder/
│   ├── autoencoder_v3.pth          ← deployed (30-epoch)
│   ├── autoencoder_v2.pth          ← 1-epoch test run
│   ├── autoencoder_v1.pth          ← recovered from interrupted run
│   └── checkpoints/
│       ├── checkpoint_epoch_001.pth
│       ├── ...
│       └── checkpoint_epoch_030.pth
├── classifier/
│   └── classifier_v3.pkl
├── anomaly/
│   └── anomaly_v3.pkl
├── similarity/
│   ├── sim_vectors.npy             ← 10,000 × 1,024 reference vectors
│   └── sim_labels.json
└── metadata/
    └── metadata.json               ← version registry
```

---

## Current Model Status

| Version | Epochs | Val MSE | Classifier Acc | Status |
|---------|--------|---------|----------------|--------|
| v1 | — | recovered | — | archived |
| v2 | 1 | 0.008246 | 45.75% | archived |
| **v3** | **30** | **0.004354** | **48.32%** | **DEPLOYED** |

> **Important:** v3 is a complete 30-epoch run. Classifier accuracy of ~48%
> is expected for a Random Forest trained on a relatively small latent space
> (1,024 dimensions) from a 30-epoch autoencoder. Performance improves further
> with longer training or a larger model.

### Verified v3 evaluation metrics

```
Deployed version : v3
Autoencoder      : FOUND
Classifier       : FOUND
Anomaly Model    : FOUND
Similarity DB    : FOUND

Mean MSE              : 0.005779
Mean PSNR             : 22.95 dB
Mean SSIM             : 0.7427
Classifier Accuracy   : ~46-48%
Compression Ratio     : 3.0 : 1 (3072 -> 1024)
```

---

## Model Versioning

The system uses a metadata-based version registry at
`models/metadata/metadata.json`.

- Every `python train.py` run creates a new version number
- The deployment gate prevents promotion unless validation MSE improves
- `python train.py --resume` resumes from the latest checkpoint on the
  same version number
- If metadata is missing (training killed before completion), the system
  **auto-recovers** the deployed version from disk files on next startup

---

## Dashboard Features

Once running at http://localhost:8501:

1. **Analyze Image tab**
   - Upload PNG / JPG / JPEG / WebP, or use a random CIFAR-10 test image
   - Side-by-side original vs reconstructed image (upscaled to 256×256)
   - Classification: predicted class, confidence, top-3 predictions
   - Compression & Quality: MSE, PSNR, SSIM, quality label
   - Anomaly Detection: score + NORMAL/ANOMALOUS badge
   - Similarity Search: top-5 cosine-similar classes from latent space
   - Full Image Specification (expandable)
   - Latent Representation heatmap (64 × 16 grid, expandable)
   - AI Explanation (evidence-based, no LLM)
   - Continuous Learning buffer button

2. **Continuous Learning tab**
   - Learning buffer status and fill progress
   - "Run Incremental Training Now" button (explicit, never automatic)
   - Deployment gate explanation

3. **Model Versions tab**
   - Full version history table
   - MSE improvement chart across versions
   - Architecture info panel

4. **About tab**
   - Architecture diagram
   - Technology stack
   - Training commands

---

## CIFAR-10 Classes

```
0: airplane    1: automobile  2: bird    3: cat    4: deer
5: dog         6: frog        7: horse   8: ship   9: truck
```

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Deep Learning | PyTorch 2.x (CPU/GPU) |
| Dataset | CIFAR-10 |
| Computer Vision | Torchvision |
| Classification | scikit-learn RandomForestClassifier |
| Anomaly Detection | scikit-learn IsolationForest |
| Metrics | MSE, PSNR, SSIM (scikit-image) |
| Similarity | Cosine similarity (NumPy) |
| Continuous Learning | Incremental fine-tuning + replay |
| Model Storage | .pth (PyTorch), .pkl (joblib) |
| Dashboard | Streamlit 1.x |
| Visualisation | Matplotlib |
| Data | NumPy, Pandas |
| Versioning | JSON metadata registry |

---

## Troubleshooting

### Training stops silently around epoch 11

**Cause:** Windows kills the process due to insufficient free RAM (< 1 GB).  
**Fix:** `python train.py --resume --epochs 30` — continues from last checkpoint.

### evaluate.py shows `Deployed version : vNone`

**Cause:** Training was killed before `register_version()` completed.  
**Fix:** `evaluate.py` auto-recovers the deployed version from disk.

### ERROR: No autoencoder found at `autoencoder_v0.pth`

**Cause:** Old bug — None deployed_version was treated as v0.  
**Status:** Fixed. The system never constructs v0 paths.

### Classifier/Anomaly NOT FOUND in evaluate.py

**Cause:** Training was interrupted before steps 5–7 (classifier/anomaly/similarity).  
**Fix:** `python train.py` — the autoencoder checkpoint exists, so the training
loop resumes correctly and only trains the downstream models.

### Streamlit crashes on startup

1. Run `python check.py` — it will identify the exact missing component.
2. Ensure `models/metadata/metadata.json` exists. If not, run `python train.py` first.
3. Check `logs/app.log` for the full traceback.

### Memory during training

- Ensure at least 1.5 GB of free RAM before starting `python train.py`.
- Close other applications (browsers, IDEs) if possible.
- Use `python train.py --resume` after a crash — no progress is lost.

---

## Expected Outputs After Full 30-Epoch Training

| Metric | Expected Range |
|--------|---------------|
| Val MSE | 0.003 – 0.008 |
| PSNR | 22 – 30 dB |
| SSIM | 0.70 – 0.90 |
| Classifier Accuracy | 45 – 65% |
| Compression Ratio | 3.0× (fixed) |

> Exact values depend on hardware and random seed.

---

## License

For academic / educational use.
