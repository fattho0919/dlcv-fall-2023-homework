# HW3: Vision-Language Models and Parameter-Efficient Tuning

This homework focuses on advanced multimodal learning techniques, exploring vision-language models and efficient fine-tuning strategies for large pre-trained models.

## 📋 Assignment Overview

### Problem 1: Image Captioning
- **Task:** Generate natural language descriptions for images
- **Model:** Vision-Transformer based encoder-decoder architecture
- **Approach:** Fine-tuning pre-trained vision-language models
- **Evaluation:** BLEU scores and human evaluation metrics

### Problem 2: Parameter-Efficient Fine-tuning
- **Task:** Efficient adaptation of large pre-trained models
- **Methods Implemented:**
  - **Adapter:** Small bottleneck modules inserted into transformer layers
  - **LoRA:** Low-Rank Adaptation for efficient parameter updates
  - **Prefix Tuning:** Learnable prefix tokens for task adaptation
- **Goal:** Achieve competitive performance with minimal parameter updates

### Problem 3: Visual Question Answering (VQA)
- **Task:** Answer questions about image content
- **Implementation:** Multimodal fusion techniques
- **Challenge:** Understanding complex visual-textual relationships

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
bash hw3_download.sh
```

## 🏃‍♂️ Usage

### Problem 1: Image Captioning
```bash
# Training
python hw3_1_train.py

# Inference
bash hw3_1.sh <test_data_path> <output_path>
```

### Problem 2: Parameter-Efficient Fine-tuning
```bash
# Adapter method
python hw3_2_adapter.py

# LoRA method
python hw3_2_lora.py

# Prefix tuning method
python hw3_2_prefix_tuning.py

# Full fine-tuning baseline
python hw3_2_finetune.py

# Inference
bash hw3_2.sh <test_data_path> <output_path>
```

### Problem 3: Advanced Tasks
```bash
# Custom implementation
python hw3_3.py
```

## 📁 File Structure

```
├── README.md                # This file
├── requirements.txt         # Dependencies
├── src/                     # Source code
│   ├── problem1/           # Image Captioning
│   │   ├── hw3_1_inference.py
│   │   └── hw3_1_train.py
│   ├── problem2/           # Parameter-Efficient Fine-tuning
│   │   ├── hw3_2_inference.py
│   │   ├── hw3_2_finetune.py
│   │   ├── hw3_2_adapter.py
│   │   ├── hw3_2_lora.py
│   │   └── hw3_2_prefix_tuning.py
│   ├── problem3/           # Advanced VQA
│   │   └── hw3_3.py
│   ├── models/             # Model implementations
│   │   ├── decoder_adapter.py
│   │   └── original_decoder.py
│   └── utils/              # Utility functions
│       ├── dataset.py
│       ├── tokenizer.py
│       └── p2_evaluate.py
├── configs/                # Configuration files
│   ├── encoder.json
│   └── vocab.bpe
├── scripts/                # Shell scripts
│   ├── hw3_1.sh
│   ├── hw3_2.sh
│   ├── get_dataset.sh
│   └── hw3_download.sh
└── results/                # Training results
    ├── *.json
    ├── *.pth
    ├── *.csv
    └── *.png
```

## 🔧 Key Features

### Parameter-Efficient Methods

#### Adapter
- **Bottleneck Architecture:** Small feedforward networks inserted into layers
- **Parameter Efficiency:** <1% of original parameters
- **Task-Specific:** Separate adapters for different tasks

#### LoRA (Low-Rank Adaptation)
- **Matrix Decomposition:** Low-rank factorization of weight updates
- **Memory Efficient:** Significant reduction in trainable parameters
- **Merge-Friendly:** Can be merged back to original weights

#### Prefix Tuning
- **Virtual Tokens:** Learnable prefix sequences
- **Context Modeling:** Enhanced context understanding
- **Non-Intrusive:** No modification to original architecture

### Vision-Language Integration
- **Multimodal Fusion:** Advanced techniques for combining visual and textual features
- **Attention Mechanisms:** Cross-modal attention for better alignment
- **Pre-trained Backbones:** Leveraging powerful vision and language models

## 📊 Results

### Image Captioning
- **BLEU Scores:** Competitive performance on standard benchmarks
- **Qualitative Analysis:** Natural and descriptive captions
- **Diversity:** Rich vocabulary and varied sentence structures

### Parameter-Efficient Fine-tuning Comparison
- **Adapter:** ~94% accuracy with 0.7% parameters
- **LoRA:** Comparable performance with minimal overhead
- **Prefix Tuning:** Effective for smaller datasets
- **Full Fine-tuning:** Baseline performance with 100% parameters

### Efficiency Analysis
- **Training Time:** Significant reduction compared to full fine-tuning
- **Memory Usage:** Lower GPU memory requirements
- **Storage:** Smaller model checkpoints for deployment

## 🛠️ Technical Implementation

- **Framework:** PyTorch, Transformers
- **Architectures:** Vision Transformer, GPT-style decoders
- **Optimization:** AdamW with cosine annealing
- **Evaluation Metrics:**
  - BLEU-1, BLEU-4 for captioning
  - Accuracy for classification tasks
  - Parameter efficiency ratios

## 🎯 Learning Outcomes

- **Multimodal Learning:** Understanding of vision-language interactions
- **Efficient Fine-tuning:** Practical techniques for large model adaptation
- **Transformer Architectures:** Deep knowledge of attention mechanisms
- **Performance Trade-offs:** Balancing efficiency and effectiveness

## 🔬 Experimental Insights

- **Parameter Efficiency:** Demonstrated that <1% of parameters can achieve >90% performance
- **Method Comparison:** Comprehensive analysis of different adaptation techniques
- **Scalability:** Efficient approaches for deploying large models in resource-constrained environments
