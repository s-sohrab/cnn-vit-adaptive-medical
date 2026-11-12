# CNN-ViT with Adaptive Attention for Medical Image Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A hybrid deep learning architecture combining **CNN** and **Vision Transformer** with a novel **Adaptive Attention** mechanism for medical image classification under limited data.

## Results on OrganAMNIST

| Model | Val Acc | Test Acc | Params | Time |
|---|---|---|---|---|
| Pure CNN | 93.76% | 80.55% | 0.1M | 0.6 min |
| Pure ViT | 95.05% | 81.47% | 0.4M | 1.6 min |
| ResNet-18 | 97.13% | 89.53% | 11.2M | 2.0 min |
| **Ours (CNN-ViT + Adaptive)** | **97.89%** | **92.71%** | **0.54M** | **0.1 min** |

## Quick Start

```bash
pip install -r requirements.txt
python src/train.py
```

## GitHub
https://github.com/s-sohrab

## HuggingFace
https://huggingface.co/smd20