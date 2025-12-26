# VAE Pokemon Generation

A Variational Autoencoder (VAE) implementation for generating Pokemon-style images. This project demonstrates VAE training, latent space interpolation, and various visualization techniques.

## 📋 Project Overview

This project implements a VAE model to learn and generate Pokemon images. The model learns a compact 16-dimensional latent representation of 40×40 RGB Pokemon sprites and can generate new images through latent space manipulation.

### Key Features

- **VAE Architecture**: Deep encoder-decoder network with reparameterization trick
- **Latent Space Exploration**: Multiple interpolation methods (linear, bilinear, spherical)
- **Training Visualization**: TensorBoard integration for monitoring training progress
- **Checkpoint System**: Save and load models at different training stages
- **Experiment Scripts**: Ready-to-use scripts for various visualization experiments

## 🏗️ Model Architecture

### Encoder
- Input: 40×40×3 RGB images (4800 dimensions)
- Hidden layers: 4-layer fully connected network (1024 units each)
- Output: 32-dimensional vector (16 for μ, 16 for log σ²)
- Activation: ReLU

### Decoder
- Input: 16-dimensional latent vector
- Hidden layers: 3-layer fully connected network (1024 units each)
- Output: 40×40×3 RGB images
- Activation: Sigmoid (final layer)

### Loss Function
```python
Total Loss = 10 × Reconstruction Loss + KL Divergence
```
- **Reconstruction Loss**: MSE between input and reconstructed images
- **KL Divergence**: Regularization to keep latent distribution close to N(0,1)

## 🚀 Getting Started

### Prerequisites

```bash
Python >= 3.8
PyTorch >= 1.8.0
torchvision
matplotlib
numpy
tensorboard
opencv-python
Pillow
```

### Installation

```bash
# Clone the repository
git clone git@github.com:Gbone3176/VAE_Pokemon.git
cd VAE_Pokemon

# Install dependencies
pip install torch torchvision matplotlib numpy tensorboard opencv-python pillow
```

### Dataset Preparation

Place your Pokemon images (PNG format, 40×40 pixels) in the `figure/` directory.

## 🎯 Usage

### Training

```bash
python main.py --hidden_size 1024 \
               --latent_size 16 \
               --batch_size 256 \
               --epoch 10000 \
               --lr 5e-4 \
               --device cuda
```

**Key Arguments:**
- `--hidden_size`: Hidden layer size (default: 1024)
- `--latent_size`: Latent vector dimension (default: 16)
- `--batch_size`: Training batch size (default: 256)
- `--epoch`: Number of training epochs (default: 10000)
- `--lr`: Learning rate (default: 5e-4)
- `--save_period`: Checkpoint saving interval (default: 200)

### Monitoring Training

```bash
tensorboard --logdir output_dir_lr0.0005_epoch10000_latent16_hidden1024
```

## 🔬 Experiments

### Experiment 1: Training Progression Visualization
Visualize how the reconstruction quality improves across different training epochs.

```bash
python vis1.py
```

**Output**: A grid showing original image and reconstructions from checkpoints at epochs 0, 200, 400, ..., 7800.

### Experiment 2: Bilinear Interpolation
Explore latent space using bilinear interpolation between four corner images.

```bash
python vis2.py
```

**Features**:
- Select 4 corner images from dataset
- Generate a 10×10 grid of interpolated images
- Smooth transitions between corners

### Experiment 3: Spherical Interpolation (SLERP)
Compare spherical linear interpolation with standard linear interpolation.

```bash
python vis3.py
```

**Note**: For VAE latent spaces (Euclidean), SLERP and LERP produce similar results.

## 📊 Training Configuration

The default training setup uses:

- **Input Size**: 40×40×3 (4800-D)
- **Latent Size**: 16-D
- **Hidden Size**: 1024-D
- **Batch Size**: 256
- **Learning Rate**: 5×10⁻⁴ with warmup and cosine decay
- **Training Split**: 99% train, 1% validation
- **Total Epochs**: 10,000
- **Optimizer**: Adam

## 📁 Project Structure

```
VAE-Pokemon-Creation/
├── main.py                 # Training script
├── model.py                # VAE model definition
├── test_vae.py             # Testing utilities
├── vis1.py                 # Time interpolation experiment
├── vis2.py                 # Bilinear interpolation experiment
├── vis3.py                 # Spherical interpolation experiment
├── utils/
│   ├── data_utils.py       # Dataset loader
│   └── lr_sched.py         # Learning rate scheduler
├── figure/                 # Dataset directory
├── vis/                    # Visualization outputs
│   ├── exp1-TimeInterpolation/
│   ├── exp2-PrarmsInterpolation/
│   └── exp3-SlerpInterpolation/
└── output_dir_*/           # Training checkpoints and logs
```

## 🎨 Key Concepts

### Reparameterization Trick
```python
z = μ + σ × ε,  where ε ~ N(0, 1)
```
This allows gradients to flow through the sampling operation during backpropagation.

### KL Divergence
Ensures the learned latent distribution q(z|x) stays close to the prior p(z) = N(0,1):
```
KL(q||p) = -0.5 × Σ(1 + log σ² - μ² - σ²)
```

### Latent Space Interpolation
- **Linear**: `z = (1-t)·z₁ + t·z₂`
- **Bilinear**: 2D interpolation using four corner points
- **Spherical**: Interpolation along great circles on hypersphere

## 📈 Results

The model achieves:
- Smooth latent space with meaningful interpolations
- High-quality reconstructions after ~5000 epochs
- Continuous generation of novel Pokemon-like sprites


## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- VAE architecture inspired by the original [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) paper
- Pokemon dataset sourced from community sprite collections

---

**Happy Generating! 🎮✨**
