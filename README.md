Adaptive Computation in CNNs: Differentiable Latency-Aware Gating (LAG)

Authors: Ahmad Mousa & Mohamad Hafiz

Lab: IoT Research Lab, Faculty of Engineering and Applied Science

Institution: Ontario Tech University

Date: November 2025

📌 Overview

This repository contains the implementation and benchmarking suite for Latency-Aware Gating (LAG), a novel framework for adaptive computation in Convolutional Neural Networks (CNNs).

Standard CNNs are computationally rigid, applying the same amount of processing power to both simple and complex inputs. LAG introduces an end-to-end differentiable mechanism that learns to turn channels ON or OFF based on the input feature map itself. This results in significant reductions in FLOPs and parameter counts without compromising accuracy.

Key Features

Input Adaptive: "Easy" images use less network capacity; "Hard" images use more.

Differentiable Soft Gating: Uses temperature-scaled sigmoid functions during training to allow gradient flow.

Hard Gating Inference: Converts to discrete ON/OFF switches during inference for real hardware speedups.

Sparsity Induction: Uses L1 regularization on gate activations to encourage pruning.

🚀 Results

Our method has been benchmarked against standard static baselines with the following highlights:

Model

Dataset

Metric

Result

VGG-11

CIFAR-100

Accuracy

+3.47% improvement over baseline

VGG-11

CIFAR-100

Parameters

25% reduction

ResNet-18

CIFAR-10

FLOPs

40% reduction (matched accuracy)

The model autonomously learns that early feature extractors are critical, while high-level semantic channels often contain redundancy.

🛠️ Installation

Clone the repository:

git clone [https://github.com/your-username/adaptive-computation-lag.git](https://github.com/your-username/adaptive-computation-lag.git)
cd adaptive-computation-lag


Install dependencies:
It is recommended to use a virtual environment.

pip install -r requirements.txt


Requirements:

torch & torchvision

matplotlib

numpy

scipy

💻 Usage

Running Benchmarks

The main entry point main.py runs a comparison suite between Baseline (static) and Gated (adaptive) models.

python main.py


This script will:

Train Baseline and Gated versions of VGG and ResNet models.

Evaluate performance on CIFAR-10, CIFAR-100, and SVHN.

Save training visualization plots to the results/ directory.

Print a comparative table to the console.

Presentation

This repository includes an interactive HTML presentation of the research.

Open index.html in your web browser to view the slides, animations, and interactive neural network demo.

📂 Project Structure

├── main.py              # Benchmark execution script
├── requirements.txt     # Python dependencies
├── index.html           # Interactive presentation slides
├── src/                 # Source code
│   ├── config.py        # Hyperparameters and configuration
│   ├── models.py        # Model definitions (VGG_Gated, ResNet_Gated, etc.)
│   ├── train.py         # Training and evaluation loops
│   └── utils.py         # Helper functions (seeding, logging)
└── results/             # Generated plots and logs


🔮 Future Work


Transformers: Applying LAG to Vision Transformers (ViT) and LLMs to dynamically prune tokens during generation.

📜 License

MIT License
