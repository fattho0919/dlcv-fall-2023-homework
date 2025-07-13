# HW1: Image Classification and Semantic Segmentation

This homework focuses on fundamental computer vision tasks: image classification and semantic segmentation. It covers both supervised learning and self-supervised learning approaches.

## 📋 Assignment Overview

### Problem 1: Image Classification
- **Task:** Multi-class image classification using ResNet architecture
- **Dataset:** Custom image dataset with data augmentation
- **Approach:** 
  - ResNet50 from scratch implementation
  - Data augmentation strategies
  - Transfer learning techniques

### Problem 2: Self-Supervised Learning
- **Task:** SSL backbone training and downstream task fine-tuning
- **Approach:**
  - Pre-training on unlabeled data
  - Fine-tuning for classification tasks
  - Feature visualization with t-SNE

### Problem 3: Semantic Segmentation
- **Task:** Satellite image segmentation using PSPNet
- **Dataset:** Satellite imagery with pixel-level annotations
- **Approach:**
  - PSPNet (Pyramid Scene Parsing Network) implementation
  - Multi-scale feature extraction
  - Mean IoU evaluation

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
bash hw1_download_ckpt.sh
```

## 🏃‍♂️ Usage

### Problem 1: Image Classification
```bash
# Training
python hw1_1a_train.py
python hw1_1b_train.py

# Inference
bash hw1_1.sh <test_data_path> <output_csv_path>
```

### Problem 2: Self-Supervised Learning
```bash
# SSL backbone training
python hw1_2ssl_train.py

# Downstream task training
python hw1_2a.py  # Different strategies
python hw1_2b.py
python hw1_2c.py
python hw1_2d.py
python hw1_2e.py

# Inference
bash hw1_2.sh <test_data_path> <output_csv_path>
```

### Problem 3: Semantic Segmentation
```bash
# Training
python hw1_3a_train.py  # For satellite images
python hw1_3b_train.py  # For mask prediction

# Inference
bash hw1_3.sh <test_data_path> <output_data_path>
```

## 📁 File Structure

```
├── README.md                  # This file
├── requirements.txt           # Dependencies
├── src/                       # Source code
│   ├── problem1/             # Image Classification
│   │   ├── hw1_1_inference.py
│   │   ├── hw1_1a_train.py
│   │   ├── hw1_1b_train.py
│   │   └── hw1_1a_val.py, hw1_1b_val.py
│   ├── problem2/             # Self-Supervised Learning
│   │   ├── hw1_2_inference.py
│   │   ├── hw1_2ssl_train.py
│   │   └── hw1_2[a-e].py
│   ├── problem3/             # Semantic Segmentation
│   │   ├── hw1_3_inference.py
│   │   ├── hw1_3a_train.py
│   │   └── hw1_3b_train.py
│   ├── models/               # Model implementations
│   │   └── pspnet.py
│   └── utils/                # Utility functions
│       ├── mean_iou_evaluate.py
│       └── viz_mask.py
├── scripts/                  # Shell scripts
│   ├── hw1_1.sh
│   ├── hw1_2.sh
│   ├── hw1_3.sh
│   ├── get_dataset.sh
│   └── hw1_download_ckpt.sh
├── checkpoint/               # Pre-trained models
├── model_data/              # Additional model files
├── hw1_data/                # Dataset
├── outputs/                 # Generated outputs
│   ├── problem3_predictions/
│   ├── problem3_masks/
│   └── testing/
└── results/                 # Training results
    ├── *.csv
    └── *.png
```

## 🔧 Key Features

- **Modular Design:** Separate training and inference scripts
- **Reproducible Results:** Fixed random seeds for consistency
- **Data Augmentation:** Comprehensive augmentation strategies
- **Visualization:** t-SNE plots and segmentation mask visualization
- **Evaluation:** Standard metrics (accuracy, mIoU)

## 📊 Results

- **Classification:** Achieved competitive accuracy on test dataset
- **SSL:** Effective feature learning demonstrated through downstream tasks
- **Segmentation:** High-quality pixel-level predictions with good mIoU scores

## 🛠️ Technical Implementation

- **Framework:** PyTorch
- **Architecture:** ResNet50, PSPNet
- **Optimization:** Adam optimizer with learning rate scheduling
- **Data Loading:** Efficient DataLoader with custom Dataset classes
- **GPU Support:** CUDA acceleration for training and inference
