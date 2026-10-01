**Full Report: [Report](Sign_Language_Time_Series_Analysis_AI-Hub_2026.pdf)**

**Download full REPORT and code and data: https://drive.google.com/drive/folders/1mkrcqbuRQQ2G0qsNYoSnPmmXH3ZWFcSR?usp=sharing**

**Sign Language Time Series Analysis Using Machine Learning Methods**

**Proxy Anchor pretraining for geometry-aware, skeleton-based isolated sign language recognition**


Author: Spyridon Bantis

**Overview**

This repository extends the geometry-aware recognition framework of in which a lightweight hand-pose
Distance Encoder is pretrained and then fused with raw skeletal features inside a
Transformer classifier.
In the original framework the encoder is pretrained with a self-supervised triplet loss
driven by geometric distances (Euclidean, Procrustes, Articulated Pose Distance).
This project replaces it with the supervised Proxy Anchor Loss (Kim et al., CVPR 2020),
so that the encoder learns from sign class labels directly, and systematically evaluates
the resulting encoder alone and fused with the geometric encoders.
```
Stage 1 – Encoder pretraining          Stage 2 – Recognition
right-hand frames (21×3)               skeleton sequence (80×225)
        │                                       │
        ▼                                       ▼
HandEmbeddingMLP 63→256→128→64   ──►   [raw features ‖ hand embedding]
trained with Proxy Anchor Loss                  │
(or triplet loss: L2 / Proc / APD)              ▼
                                       Transformer encoder (4 layers, 8 heads)
                                                │
                                                ▼
                                       avg pooling → linear → sign class
```
Benchmarks: WLASL-100, WLASL-300, WLASL-2000, using MediaPipe Holistic skeletons.
---
Repository structure
```
.
├── config.yaml                          # dataset paths and pretraining settings
├── experiment_B_proxy_anchor_pretrain.py  # Proxy Anchor encoder pretraining  
├── demo_v2.py                           # qualitative demonstration tool 
├── src/
│   ├── pretraining/
│   │   ├── 1_distance_to_triplets.py    # geometric distances → triplets
│   │   ├── 2_distance_encoder.py        # triplet-loss encoder pretraining
│   │   └── embeding_networks.py
│   ├── training/
│   │   ├── train.py                     # Transformer recognition training
│   │   ├── models.py
│   │   └── dataset.py
│   └── utils/
├── data/                                # (not included) WLASL .npy splits
├── outputs/                             # pretrained encoders
└── logs/                                # one folder per training run
```
---
1. Installation
Tested with Python 3.11, PyTorch with CUDA, on Linux and Windows.
```bash
conda create -n project python=3.11 -y
conda activate project
pip install torch numpy pyyaml tqdm scipy scikit-learn einops matplotlib
```
The code imports modules as `src.…`, so the repository root must be on `PYTHONPATH`:
```bash
# Linux / macOS
export PYTHONPATH=$(pwd)

# Windows PowerShell
$env:PYTHONPATH = (Get-Location).Path
```
Run every command below from the repository root.
---
2. Data
The WLASL data are not included in this repository. WLASL videos are available in the google drive above or from the
official WLASL project under its own license.
Skeletons are extracted with MediaPipe Holistic and stored as NumPy arrays:
Item	Shape	Description
`X_{train,val,test}.npy`	`(N, 80, 225)`	80 temporally normalized frames × 75 joints × (x, y, z)
`y_{train,val,test}.npy`	`(N,)`	integer class labels
Joint order (75 joints): 33 body pose joints (0–32), 21 left-hand joints (33–53),
21 right-hand joints (54–74).
Expected layout:
```
data/
├── WLASL_npy_dataset_100_split/
├── WLASL_npy_dataset_300_split/
└── WLASL_npy_dataset_2000_split/
    ├── X_train.npy  y_train.npy
    ├── X_val.npy    y_val.npy
    └── X_test.npy   y_test.npy
```
Subset	Classes	Train	Val	Test
WLASL-100	100	1,408	329	249
WLASL-300	300	3,459	887	648
WLASL-2000	2,000	14,061	3,890	2,848
---
3. Configuration
Before working on a subset, point `config.yaml` to it:
```yaml
data:
  path: data/WLASL_npy_dataset_2000_split
  output_path: outputs/WLASL_npy_dataset_2000_split/

triplet_generation:
  num_triplets: 200000
  subsampling: 0.1      # fraction of right-hand frames kept for distance computation
  pos_k: 30
  neg_start: 50
  neg_end: 100
```
---
4. Stage 1a — Geometric encoders (triplet loss, baseline)
For each metric, compute pairwise distances and triplets, then train the encoder (20 epochs):
```bash
# Euclidean
python src/pretraining/1_distance_to_triplets.py --metric L2 --hand right
python src/pretraining/2_distance_encoder.py --path-extension L2_r

# Rotation-regularized Procrustes
python src/pretraining/1_distance_to_triplets.py --metric Proc-weighted --hand right
python src/pretraining/2_distance_encoder.py --path-extension Proc-weighted_r

# Articulated Pose Distance (quaternions)
python src/pretraining/1_distance_to_triplets.py --metric Quat --hand right
python src/pretraining/2_distance_encoder.py --path-extension Quat_r
```
Encoders are saved to `outputs/<subset>/<metric>_r/dist_encoder.pt`.
> **Memory and runtime notes (WLASL-2000).** The full pairwise distance matrix is large.
> For `Quat` on WLASL-2000 we used `subsampling: 0.05` and reduced `batch_size` in
> `pairwise_distances()` (in `1_distance_to_triplets.py`) from 256 to 64 to avoid running
> out of RAM. On an RTX 4090 with 62 GB RAM, triplet generation took about 4 min (L2),
> 3 h (Procrustes) and 16.5 h (Quat). Run long jobs inside `tmux` or `screen`.
---
5. Stage 1b — Proxy Anchor encoder 
```bash
python experiment_B_proxy_anchor_pretrain.py \
    --data_dir data/WLASL_npy_dataset_2000_split \
    --save_path outputs/proxy_encoder_2000.pt \
    --epochs 50
```
Defaults: embedding dimension 64, α = 32, δ = 0.1, AdamW, cosine schedule.
Run `python experiment_B_proxy_anchor_pretrain.py --help` for all options.
---
6. Stage 2 — Recognition model
All configurations share the same Transformer settings:
```bash
COMMON="--seed 1 --epochs 50 --batch_size 32 --lr 3e-4 --wd 1e-4 \
        --heads 8 --layers 4 --model_dim 256 --use_posenc"
D=data/WLASL_npy_dataset_2000_split
E=outputs/WLASL_npy_dataset_2000_split
```
Baseline without encoder
```bash
python src/training/train.py --data_dir $D --no_hand_encoder $COMMON
```
Single geometric encoder (example: APD)
```bash
python src/training/train.py --data_dir $D --encoder_ckpt $E/Quat_r/dist_encoder.pt $COMMON
```
Proxy Anchor encoder, fine-tuned
```bash
python src/training/train.py --data_dir $D --encoder_ckpt outputs/proxy_encoder_2000.pt \
    --hand_emb_dim 64 $COMMON
```
Add `--freeze_encoder` for the frozen variant.
Geometric fusion (Proc + L2 + APD)
```bash
python src/training/train.py --data_dir $D \
    --encoder_ckpt $E/Quat_r/dist_encoder.pt $E/Proc-weighted_r/dist_encoder.pt $E/L2_r/dist_encoder.pt \
    --encoder_fusion weighted $COMMON
```
Geometric fusion + Proxy Anchor
```bash
python src/training/train.py --data_dir $D \
    --encoder_ckpt $E/Quat_r/dist_encoder.pt $E/Proc-weighted_r/dist_encoder.pt $E/L2_r/dist_encoder.pt \
                   outputs/proxy_encoder_2000.pt \
    --encoder_fusion weighted $COMMON
```
Use `--encoder_fusion concat` for concatenated fusion.
Each run creates `logs/<timestamp>/` containing `best.pt` (best validation checkpoint),
`encoders.json` (encoder configuration) and `train.log`. The final line of `train.log`
reports test Top-1 / Top-5 / Top-10 accuracy.
On Windows, replace the shell variables with full paths and put each command on one line.
---
7. Hyperformer (optional)
`experiment_C_hyperformer_wlasl.py` adapts the Hypergraph Transformer to the 75-joint
MediaPipe skeleton (custom graph and 8-group joint partition), with optional auxiliary
Proxy Anchor Loss. It requires a GPU with ~16 GB VRAM (we used a Colab T4).
```bash
python experiment_C_hyperformer_wlasl.py --help
```
---
8. Demonstration tool
`demo_v2.py` loads a trained model and, for a test sample, shows three panels:
the test skeleton animation, a training example of the predicted class, and the Top-5
predictions.
Set the paths at the top of the script to your own run:
```python
DATA_DIR = "data/WLASL_npy_dataset_100_split"
CKPT = "logs/<your_run>/best.pt"
ENCODER_CKPT = ["outputs/proxy_encoder.pt"]
```
```bash
python demo_v2.py                  # random test sample
python demo_v2.py --index 42       # specific sample
python demo_v2.py --num_samples 5  # several in a row (close the window to advance)
python demo_v2.py --save_gif       # save the animation as a GIF
```
The model settings in `load_model()` must match the run being loaded
(see that run's `encoders.json`).
---
9. Results (Top-1 accuracy, %, single run, seed 1)
Configuration	WLASL-100	WLASL-300	WLASL-2000
No embedding	69.5	–	26.3
Proc + L2 + APD (weighted)	76.5	64.4	36.4
Proxy Anchor (fine-tuned)	78.3	59.3	35.3
Proc + L2 + APD + Proxy (weighted)	77.1	64.8	36.1
Proc + L2 + APD + Proxy (concat)	77.5	63.3	35.5
On WLASL-2000, adding Proxy Anchor to the geometric fusion leaves Top-1 essentially
unchanged but improves Top-5 from 66.9 to 69.5 and Top-10 from 76.8 to 78.8.
Full tables are in the report.
Caveat: all numbers come from a single run per configuration, so differences of about
1–2 points are within run-to-run variance.
---
Hardware used
Experiments	GPU
WLASL-100 / 300	NVIDIA RTX 3050 (4 GB)
WLASL-2000	NVIDIA RTX 4090 (24 GB), University of Patras CEID computing centre
Hyperformer	NVIDIA T4 (16 GB), Google Colab
---
Citation
If you use this work, please cite the original framework and the methods it builds on:
```bibtex
@inproceedings{sartinas2026geometric,
  title     = {Geometric Skeletal Distance Learning for Self-Supervised Sign Language Recognition},
  author    = {Sartinas, Evangelos G. and Kosmopoulos, Dimitrios and Psarakis, Emmanouil Z.
               and Blekos, Kostas and De, Bikram Kumar and Metsis, Vangelis},
  booktitle = {VISAPP},
  year      = {2026}
}

@inproceedings{kim2020proxy,
  title     = {Proxy Anchor Loss for Deep Metric Learning},
  author    = {Kim, Sungyeon and Kim, Dongwon and Cho, Minsu and Kwak, Suha},
  booktitle = {CVPR},
  year      = {2020}
}

@article{zhou2022hyperformer,
  title   = {Hypergraph Transformer for Skeleton-based Action Recognition},
  author  = {Zhou, Yuxuan and Cheng, Zhi-Qi and Li, Chao and Fang, Yanwen and Geng, Yifeng
             and Xie, Xuansong and Keuper, Margret},
  journal = {arXiv preprint arXiv:2211.09590},
  year    = {2022}
}
```
