# ViMbaFormer  
### A Hybrid Vision–Mamba U-Net Architecture for UAV Semantic Segmentation

ViMbaFormer is a lightweight hybrid semantic segmentation framework that integrates **Vision Transformers**, **State Space Models (Mamba)**, and a **U-Net–style encoder–decoder architecture** for efficient UAV and aerial image understanding.  
The model is designed to capture both **local spatial features** and **long-range global dependencies** while maintaining low computational cost.

This repository provides the official PyTorch implementation of ViMbaFormer.

---

## 🚀 Key Features

- Hybrid CNN + Transformer + Mamba architecture  
- Efficient long-range modeling using State Space Models  
- U-Net inspired encoder–decoder with skip connections  
- Adaptive feature fusion and alignment blocks  
- Lightweight design suitable for UAV / remote sensing applications  
- Supports multi-class semantic segmentation  

---

## 🧠 Architecture Overview

ViMbaFormer consists of:

- **Encoder**: Lightweight CNN backbone with hierarchical feature extraction  
- **Bottleneck**: Vision Transformer + Mamba blocks for global context modeling  
- **Decoder**: Adaptive Fusion + AlignBlocks for multi-scale feature reconstruction  
- **Segmentation Head**: Produces pixel-wise class predictions  

The network effectively balances:

- Local texture understanding (CNN)  
- Global dependency modeling (Transformer)  
- Linear-complexity sequence processing (Mamba)

---


---

## 📊 Datasets

The model is designed for UAV / aerial semantic segmentation datasets such as:

- UAVid  
- ISPRS Potsdam / Vaihingen  
- Custom aerial imagery  

Prepare datasets in the following format:

dataset/
├── images/
└── masks/


---

## 🛠 Installation

```bash
git clone https://github.com/yourusername/ViMbaFormer.git
cd ViMbaFormer

pip install -r requirements.txt
```

## 📁 Project Structure

