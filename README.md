

# Adaptive Computation in CNNs via Differentiable Latency-Aware Gating

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)](https://pytorch.org/)

> **Authors:** Ahmad Mousa & Mohamad Hafiz  
> **Lab:** IoT Research Lab, University of Ontario Institute of Technology

---

## 📌 Overview
This repository contains the official implementation of our research paper: **"Adaptive Computation in Convolutional Neural Networks via Differentiable Latency-Aware Gating"**.

We introduce a **dynamic channel pruning mechanism** that allows CNNs to automatically learn optimal sub-architectures during training. By using a temperature-scaled sigmoid gate and a latency-regularized loss function, our method achieves significant parameter reduction without sacrificing accuracy.

### Key Features
* **🔥 Differentiable Gating:** End-to-end optimization of channel selection using soft gates.
* **📉 Latency-Aware Loss:** Explicitly penalizes computational cost (FLOPs proxy).
* **🌡️ Temperature Annealing:** Smooth transition from soft attention to hard pruning.
* **📊 Comprehensive Benchmarks:** Validated on VGG-11, ResNet-18, and ResNet-34 across CIFAR-10, CIFAR-100, and SVHN.

---

## 🧪 Experimental Results

Our method demonstrates that deep networks are often over-parameterized for standard tasks. We consistently reduce model size while maintaining (or even improving) accuracy.

| Model | Dataset | Baseline Accuracy | **Gated Accuracy** | **Parameters Pruned** |
| :--- | :--- | :---: | :---: | :---: |
| **VGG-11** | CIFAR-100 | 91.38% | **94.85% (+3.47%)** | **24.8%** |
| **ResNet-18** | CIFAR-10 | 98.94% | 98.93% (-0.01%) | **25.2%** |
| **ResNet-34** | SVHN | 100.00% | 100.00% (+0.00%) | **36.2%** |

### Visualizations
| Training Dynamics | Pruning Efficiency | Accuracy Delta |
| :---: | :---: | :---: |
| <img src="figures/training_dynamics.png" width="300"> | <img src="figures/performance_summary.png" width="300"> | <img src="figures/accuracy_delta.png" width="300"> |

* **Left:** VGG-11 accuracy rises even as 25% of the network is pruned.
* **Center:** Simpler tasks (SVHN) allow for deeper pruning (36%).
* **Right:** The Gated VGG-11 acts as a regularizer, boosting performance on CIFAR-100.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* PyTorch 2.0+
* Torchvision
* Matplotlib, Numpy, Scipy

git clone [https://github.com/yourusername/latency-aware-gating.git](https://github.com/yourusername/latency-aware-gating.git)
cd latency-aware-gating
pip install -r requirements.txt
