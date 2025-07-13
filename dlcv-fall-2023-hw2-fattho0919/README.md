# HW2: Domain Adaptation and Diffusion Models

This homework explores advanced topics in computer vision including domain adaptation for cross-domain classification and generative modeling using diffusion models.

## 📋 Assignment Overview

### Problem 1: Digit Classification with Domain Adaptation
- **Task:** Cross-domain digit recognition (MNIST → SVHN, MNIST → USPS)
- **Challenge:** Domain shift between different digit datasets
- **Approach:** Domain Adversarial Neural Networks (DANN) implementation
- **Evaluation:** Classification accuracy on target domain

### Problem 2: Diffusion Models for Image Generation
- **Task:** Image generation using DDPM and DDIM
- **Implementation:** UNet-based denoising network
- **Features:**
  - DDPM (Denoising Diffusion Probabilistic Models)
  - DDIM (Denoising Diffusion Implicit Models)
  - Controllable generation with different eta values
  - Image interpolation capabilities

### Problem 3: Domain Adaptation Analysis
- **Task:** Comprehensive evaluation of domain adaptation methods
- **Baselines:** 
  - Lower bound (target domain only)
  - Upper bound (source domain only)
  - DANN (domain adversarial approach)
- **Visualization:** t-SNE plots for domain and class distributions

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Download Dataset
```bash
bash get_dataset.sh
```

### Download Pre-trained Models
```bash
bash hw2_download.sh
```

## 🏃‍♂️ Usage

### Problem 1: Digit Classification
```bash
# Training
python hw2_1_train.py

# Inference
bash hw2_1.sh <test_data_path> <output_csv_path>
```

### Problem 2: Diffusion Models
```bash
# Training
python hw2_2_train.py  # (if training script exists)

# Inference (Generation)
bash hw2_2.sh <output_dir>

# Interpolation
python hw2_2_interpolation.py
```

### Problem 3: Domain Adaptation
```bash
# SVHN experiments
python hw2_3_svhn_Lowerbound_train.py
python hw2_3_svhn_Upperbound_train.py
python hw2_3_svhn_DANN_train.py

# USPS experiments
python hw2_3_usps_Lowerbound_train.py
python hw2_3_usps_Upperbound_train.py
python hw2_3_usps_DANN_train.py

# Inference
bash hw2_3.sh <target_domain> <test_data_path> <output_csv_path>
```

## 📁 File Structure

```
├── digit_classifier.py        # Base classifier implementation
├── DANN.py                   # Domain Adversarial Neural Network
├── UNet.py                   # UNet architecture for diffusion
├── DDPMnDDIM.py             # Diffusion models implementation
├── hw2_1_*.py               # Problem 1 scripts
├── hw2_2_*.py               # Problem 2 scripts
├── hw2_3_*.py               # Problem 3 scripts
├── utils.py                 # Utility functions
└── requirements.txt         # Dependencies
```

## 🔧 Key Features

### Domain Adaptation (DANN)
- **Gradient Reversal Layer:** Adversarial training for domain-invariant features
- **Feature Extractor:** Shared representation learning
- **Domain Classifier:** Discriminator for domain adaptation
- **Class Classifier:** Task-specific prediction head

### Diffusion Models
- **UNet Architecture:** Advanced denoising network with attention
- **Noise Scheduling:** Linear and cosine noise schedules
- **Sampling Strategies:** DDPM and DDIM sampling
- **Controllable Generation:** Different eta values for generation control

### Visualization & Analysis
- **t-SNE Plots:** Domain and class distribution visualization
- **Training Curves:** Loss and accuracy monitoring
- **Generated Samples:** Quality assessment of generated images

## 📊 Results

### Domain Adaptation
- **SVHN Adaptation:** Improved cross-domain accuracy using DANN
- **USPS Adaptation:** Effective feature alignment across domains
- **Visualization:** Clear domain separation and class clustering in t-SNE

### Diffusion Models
- **Image Quality:** High-fidelity image generation
- **Controllability:** Various generation styles with different eta values
- **Interpolation:** Smooth transitions between images

## 🛠️ Technical Implementation

- **Framework:** PyTorch
- **Architectures:** ResNet, UNet, DANN
- **Optimization:** Adam optimizer with domain adaptation strategies
- **Loss Functions:** 
  - Classification loss
  - Domain adversarial loss
  - Diffusion denoising loss
- **Evaluation Metrics:** Classification accuracy, FID score

## 🎯 Learning Outcomes

- Understanding of domain adaptation challenges and solutions
- Implementation of state-of-the-art generative models
- Experience with adversarial training techniques
- Practical knowledge of diffusion model architectures
