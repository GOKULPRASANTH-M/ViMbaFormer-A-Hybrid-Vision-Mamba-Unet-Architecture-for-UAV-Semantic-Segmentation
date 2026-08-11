# ViMbaFormer  
## A Hybrid Vision–Mamba U-Net Architecture for UAV Semantic Segmentation

ViMbaFormer is a hybrid semantic segmentation framework designed for high-resolution UAV imagery. It integrates **Vision Transformers**, **Mamba State Space Models**, and a **U-Net–style encoder–decoder** to jointly model global context and fine spatial details with linear-time complexity.

The architecture is tailored for aerial scene understanding, urban analysis, and autonomous UAV applications.

---

## 🚀 Highlights

- Hybrid CNN + Transformer + Mamba design  
- Visual State Space Blocks (VSSB) for long-range dependency modeling  
- Hybrid Attention (Mamba + Window Attention + Convolution)  
- Adaptive feature fusion decoder with boundary refinement  
- Lightweight and efficient for large UAV images  
- Strong performance on UAVid, UDD6, and SDD benchmarks  

---

## 🧠 Architecture Overview

ViMbaFormer consists of:

### Encoder
- CNN backbone for hierarchical spatial feature extraction  
- Multiple **Visual State Space Blocks (VSSB)** for global receptive fields  

### Bottleneck
- Hybrid attention combining:
  - Mamba selective state spaces (global context)
  - Local window attention
  - Convolutional spatial refinement  

### Decoder
- U-Net style upsampling  
- Adaptive fusion with learnable weights  
- Boundary-aware attention modules  

### Segmentation Head
- Produces pixel-wise class predictions.

This design balances:

- Local texture (CNN)  
- Global dependencies (Mamba + attention)  
- Efficient computation (linear-complexity SSMs)
  
---

## 📊 Supported Datasets

- UAVid (8 classes)  
- UDD6 (6 classes)  
- SDD (22 classes)  

### Dataset Format

```text
dataset/
├── images/
└── masks/
```

Images are padded and split into **1024×1024 patches**:

- UAVid → 8 patches / image  
- UDD6 → 6 patches / image  
- SDD → 24 patches / image  

This preserves full spatial information while enabling efficient training.

---
## 🛠 Installation
### 1. Clone the repository
```bash
git clone https://github.com/GOKULPRASANTH-M/ViMbaFormer-A-Hybrid-Vision-Mamba-Unet-Architecture-for-UAV-Semantic-Segmentation.git

cd ViMbaFormer-A-Hybrid-Vision-Mamba-Unet-Architecture-for-UAV-Semantic-Segmentation
```
### 2. Install dependencies
```bash
pip install -r semantic_segmentation/requirements.txt
```
### 3. Install the selective scan module
```bash
cd selective_scan/selective_scan

pip install -e .

cd ../..
```

## ▶️ Training

```bash
cd semantic_segmentation
python train_supervision.py
python train.py --config semantic_segmentation/config.yaml
```

Default settings:
- Optimizer: AdamW
- LR: 6e-5 (cosine annealing)
- Patch size: 1024×1024
- Batch size: 2 (train), 8 (val)
- Loss: Cross-Entropy + Dice + Auxiliary supervision

## 🧪 Testing
```bash
python test.py --checkpoint checkpoints/vimbaformer.pth
```

## 📈 Results
### 📊 Benchmark Results

- **UAVid**  
  - mIoU: **69.5%**

- **UDD6**  
  - mIoU: **79.9%**

- **SDD**  
  - mIoU: **74.1%**  
  - F1-score: **83.0%**

ViMbaFormer consistently outperforms CNN, Transformer, and prior Mamba baselines, especially on:
Fine structures
Small objects
Urban boundaries

## ⚙️ Model Complexity
| Model        | mIoU (UAVid) | Params (M) | MACs (G) |
|-------------|-------------|-----------|----------|
| LAPNet-L    | 66.0        | 8.6       | 25.4     |
| UNetFormer | 67.8        | 17.7      | 62.1     |
| **ViMbaFormer** | **69.5** | **19.0** | **68.4** |


## 🔬 Key Contributions
Hybrid attention combining Mamba + window attention + convolution
Visual State Space Blocks for linear-complexity global modeling
Adaptive decoder with learnable multi-scale fusion
Strong generalization across multiple UAV datasets

## Acknowledgements

This repository includes a modified version of the selective scan
implementation derived from the Mamba project by Albert Gu and Tri Dao:

https://github.com/state-spaces/mamba

The original copyright notices are retained in the corresponding
source files. Please refer to the applicable license in the original
Mamba repository for the terms governing the selective scan code.

Modifications and integration for this project are by Gokulprasanth Mahalingam.

## 📜 Citation
If you use this work, please cite:
```bash
@article{vimbaformer2026,
  title={ViMbaFormer: A Hybrid Vision–Mamba U-Net Architecture for UAV Semantic Segmentation},
  author={Mahalingam, Gokulprasanth and Deepak Kumar, B.S. and Vishnupriya, G. and Vasamsetti, Srikanth},
  year={2026}
}
```
