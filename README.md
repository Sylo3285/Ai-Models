# 🧠 AI Models Repository

A comprehensive collection of practical implementations for various neural network architectures. This repository demonstrates how to build, train, and deploy different types of AI models using PyTorch.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Overview

This repository contains hands-on implementations of major deep learning architectures, from foundational models to state-of-the-art approaches. Each architecture is implemented with complete training pipelines, inference scripts, and detailed documentation.

## 🗂️ Repository Structure

```
Ai-Models/
├── ANN_MLP/              # Artificial Neural Networks / Multi-Layer Perceptrons
├── CNN/                  # Convolutional Neural Networks
├── RNN_LSTM_GRU/         # Recurrent Neural Networks
├── Transformers/         # Transformer-based Models
├── Autoencoders/         # Autoencoder Architectures
├── GANs/                 # Generative Adversarial Networks
├── Diffusion_Models/     # Diffusion Models
├── GNNs/                 # Graph Neural Networks
└── SNNs/                 # Spiking Neural Networks
```

## 🔍 Architecture Overview

### 🟢 Beginner Level

**ANN/MLP** - Fully-connected feedforward networks for general-purpose learning tasks. Best for tabular data, classification, and regression where spatial/temporal structure isn't critical.

**CNN** - Specialized for grid-structured data with spatial hierarchies. Excels at computer vision tasks like image classification, object detection, and feature extraction through convolutional filters.

### 🟡 Intermediate Level

**Autoencoders** - Unsupervised learning for dimensionality reduction and representation learning. Compresses input to latent space and reconstructs it, useful for denoising, compression, and anomaly detection.

**RNN/LSTM/GRU** - Designed for sequential and temporal data processing. Handles variable-length sequences with memory mechanisms, ideal for time series, language modeling, and sequential prediction.

### 🟠 Advanced Level

**Transformers** - Attention-based architecture that processes sequences in parallel. State-of-the-art for NLP, machine translation, and any task requiring long-range dependencies without recurrence.

**GANs** - Adversarial framework with generator and discriminator networks. Creates realistic synthetic data through competitive training, excelling at image generation and style transfer.

**GNNs** - Process graph-structured data with nodes and edges. Learns representations by aggregating neighbor information, perfect for social networks, molecules, and recommendation systems.

### 🔴 Expert Level

**Diffusion Models** - Iterative denoising process for high-quality generation. Gradually removes noise to create samples, achieving state-of-the-art results in image synthesis and conditional generation.

**SNNs** - Bio-inspired networks using spike-based communication. Mimics biological neurons for energy-efficient computing, targeting neuromorphic hardware and event-driven processing.

## 🚀 Neural Network Architectures

> **💡 Learning Tip:** Architectures are organized by difficulty level. Start with beginner-level architectures and progress through each level to build a strong foundation!

---

## 🟢 BEGINNER LEVEL

### 🔷 ANN/MLP - Artificial Neural Networks
**Difficulty:** ⭐ Beginner | **Status:** ✅ Complete  
**Use Cases:** Classification, Regression, Feature Learning

**The foundation of deep learning!** Start here to understand how neural networks work.

Fully-connected feedforward networks with advanced training features:
- BPE tokenization for text processing
- Learning rate scheduling & early stopping
- Comprehensive metrics tracking
- **Example Project:** Emotion classification from text

[📖 View ANN/MLP Documentation](./ANN_MLP/README.md)

---

### 🔷 CNN - Convolutional Neural Networks
**Difficulty:** ⭐ Beginner | **Status:** 🚧 In Development  
**Use Cases:** Image Classification, Object Detection, Computer Vision

**Learn spatial feature extraction** - the backbone of computer vision.

Specialized architectures for processing grid-like data:
- 2D/3D convolutions
- Pooling layers
- Batch normalization
- **Planned Projects:** Image classification, object detection

---

## 🟡 INTERMEDIATE LEVEL

### 🔷 Autoencoders
**Difficulty:** ⭐⭐ Intermediate | **Status:** 🚧 Planned  
**Use Cases:** Dimensionality Reduction, Denoising, Feature Learning

**Master unsupervised learning** for compression and representation learning.

Unsupervised learning architectures:
- Vanilla Autoencoders
- Variational Autoencoders (VAE)
- Denoising Autoencoders
- **Planned Projects:** Image compression, anomaly detection

---

### 🔷 RNN/LSTM/GRU - Recurrent Neural Networks
**Difficulty:** ⭐⭐ Intermediate | **Status:** 🚧 In Development  
**Use Cases:** Sequence Modeling, Time Series, NLP

**Handle sequential data with memory** - understand temporal dependencies.

Networks designed for sequential data:
- Vanilla RNN
- Long Short-Term Memory (LSTM)
- Gated Recurrent Units (GRU)
- **Planned Projects:** Time series forecasting, text generation

---

## 🟠 ADVANCED LEVEL

### 🔷 Transformers
**Difficulty:** ⭐⭐⭐ Advanced | **Status:** ✅ Complete  
**Use Cases:** NLP, Machine Translation, Sequence-to-Sequence Tasks

**State-of-the-art attention mechanisms** - powers ChatGPT, BERT, and modern LLMs.

Attention-based architectures for sequence processing:
- **Encoder-Decoder:** Full transformer with encoder and decoder stacks
- **Decoder-Only:** GPT-style autoregressive models
- Multi-head self-attention mechanisms
- Positional encoding

[📖 View Transformers Documentation](./Transformers/)

---

### 🔷 GANs - Generative Adversarial Networks
**Difficulty:** ⭐⭐⭐ Advanced | **Status:** 🚧 Planned  
**Use Cases:** Image Generation, Style Transfer, Data Augmentation

**Adversarial training** - tricky to train but incredibly powerful for generation.

Adversarial training frameworks:
- Vanilla GAN
- DCGAN (Deep Convolutional GAN)
- StyleGAN
- **Planned Projects:** Image generation, super-resolution

---

### 🔷 GNNs - Graph Neural Networks
**Difficulty:** ⭐⭐⭐ Advanced | **Status:** 🚧 Planned  
**Use Cases:** Social Networks, Molecular Property Prediction, Recommendation Systems

**Graph-based learning** for non-Euclidean data structures.

Networks for graph-structured data:
- Graph Convolutional Networks (GCN)
- Graph Attention Networks (GAT)
- GraphSAGE
- **Planned Projects:** Node classification, link prediction

---

## 🔴 EXPERT LEVEL

### 🔷 Diffusion Models
**Difficulty:** ⭐⭐⭐⭐ Expert | **Status:** 🚧 Planned  
**Use Cases:** Image Generation, Denoising, Inpainting

**Cutting-edge generation** - the tech behind DALL-E and Stable Diffusion.

State-of-the-art generative models:
- DDPM (Denoising Diffusion Probabilistic Models)
- Latent Diffusion Models
- **Planned Projects:** Text-to-image generation

---

### 🔷 SNNs - Spiking Neural Networks
**Difficulty:** ⭐⭐⭐⭐ Expert | **Status:** 🚧 Planned  
**Use Cases:** Neuromorphic Computing, Event-Based Vision, Energy-Efficient AI

**Bio-inspired spike-based networks** - cutting-edge research mimicking biological neurons.

Brain-inspired computing models:
- Leaky Integrate-and-Fire (LIF) neurons
- Spike-timing-dependent plasticity (STDP)
- **Planned Projects:** Event-based classification

---

## 🎯 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Install PyTorch (visit pytorch.org for your specific setup)
pip install torch torchvision torchaudio

# Common dependencies
pip install numpy pandas matplotlib tqdm scikit-learn
```

### Running a Model

Each architecture folder contains its own implementation. Navigate to the specific folder and follow its README:

```bash
# Example: Running the ANN/MLP emotion classifier
cd ANN_MLP
pip install -r requirements.txt
python datasetmaker.py  # Prepare dataset
python train.py         # Train model
python inference.py     # Run inference
```

## 📚 Learning Path

**Recommended order for beginners:**

1. **ANN/MLP** - Start here to understand basic neural networks
2. **CNN** - Learn spatial feature extraction
3. **RNN/LSTM/GRU** - Understand sequential processing
4. **Transformers** - Master attention mechanisms
5. **Autoencoders** - Explore unsupervised learning
6. **GANs** - Dive into generative models
7. **Diffusion Models** - State-of-the-art generation
8. **GNNs** - Graph-structured data
9. **SNNs** - Advanced neuromorphic computing

## 🛠️ Common Features

All implementations include:

- ✅ **Clean, documented code** with type hints
- ✅ **Configurable hyperparameters** via config files
- ✅ **Training scripts** with metrics tracking
- ✅ **Inference scripts** for deployment
- ✅ **GPU acceleration** support (CUDA)
- ✅ **Model checkpointing** and saving
- ✅ **Visualization tools** for metrics and results

## 📊 Hardware Requirements

- **Minimum:** CPU with 8GB RAM
- **Recommended:** NVIDIA GPU with 8GB+ VRAM
- **Optimal:** NVIDIA GPU with 16GB+ VRAM for larger models

GPU acceleration is automatically detected and used when available.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new model implementations
- Improve existing code
- Fix bugs or add features
- Enhance documentation

## 📝 License

This repository is for educational purposes. Individual projects may have their own licenses.

## 🔗 Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [Papers with Code](https://paperswithcode.com/)
- [Hugging Face](https://huggingface.co/)

## 📧 Contact

For questions or suggestions, please open an issue in this repository.

---

**Last Updated:** December 2025  
**Maintained by:** Sylo
