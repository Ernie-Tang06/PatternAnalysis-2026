# COMP3710 Pattern Recognition and Analysis Final Project

**Project 2.1:** Classification: Medical Diagnosis and Social Network Analysis

**Student Name:** Soong Earn Tang

**Student ID:** 50126558

**Model:** ConvNeXt

**Dataset:** ADNI


Alzheimer's disease (AD) is a progressive neurodegenerative disorder that affects memory and cognitive function. This project explores ConvNeXt-Tiny for binary classification of 2D brain MRI slices from the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset, distinguishing Alzheimer's disease (AD) from normal controls (NC).

Inspired by Vision Transformer design principles, ConvNeXt modernizes convolutional neural networks using larger kernels, Layer Normalization, GELU activation, and inverted bottleneck blocks. In this project, the model is adapted for single-channel MRI images and two-class output, implemented in PyTorch, and trained from scratch.

The objective is to evaluate the effectiveness of ConvNeXt-Tiny for MRI-based Alzheimer's disease classification using metrics such as accuracy, precision, recall, and F1-score. This research explores the potential of modern CNN architectures for computer-aided neuroimaging analysis.


---

# Table of Contents

1. Feasibility Review
2. Purpose of each file
   - `dataset.py`
   - `modules.py`
   - `train.py`
   - `predict.py`
3. Comparison between 2 models
4. Tuning model to reach 0.8 test accuracy
5. Final Setup
6. Results
7. Fault Analysis


---

# Feasibility Review (15/10/2026)

## 1. User Need, Scope & Acceptance Criteria

**Intended User:** healthcare professionals like doctors, nurses, other medical staff and patients and their families.

**Prototype Purpose:** detect this disease at an early stage as it is irreversible. Once it is detected, it can be treated earlier.

### Acceptance Criteria

1. Minimum **0.8 test accuracy** on AD vs CN.

2. It is better to have more false positives compare to false negatives.

If there is a false positive, further testing can prove that the person is healthy. However, if there is a false negative, the person may be declared healthy and miss the best treatment time.

3. It must be leakage-free evaluation.

**Peak VRAM Constraint:** personal GPU (NVIDIA GeForce RTX 4050 Laptop) = 6GB


## 2. Model Choice & Course Concepts

**Model Choice:** ConvNeXt

**Dataset Configuration:** ADNI dataset (split into test and train dataset accordingly and each with AD and NC data inside)

### Model Architecture

![ConvNeXt Architecture](media/image1.png)

Unlike most modern models which uses transformers, ConvNeXt uses pure convolution. However, it is modernized by changing several things such as big kernel size, layer normalization, GELU and others.

**Difficulty Band:** Hard

### How course concepts and literature support the choices

CNN was introduced at very early stages in this course.

ConvNeXt was shown to outperform certain transformers such as Swin Transformers by modernizing normal convolutions (*A ConvNet for the 2020s*, Liu et al., 2022).


## 3. Preliminary Feasibility Evidence

- **20 epochs:** 1 hour
- **Test accuracy:** 0.71 (way lower than required 0.8)

![Training Curve](media/image2.png)

**Resource estimate:** 2 hours per train (increase in epoch and other changes)


## 4. Risks, Budget & Fallback

**Key technical risks:**
- Data leakage between train and test
- Overfitting
- Training instability

**Computing budget:** peak VRAM 6GB

**Next planned experiment:** increasing number of epochs is not very reasonable, so other factors need tuning such as hyperparameters, data preprocessing, data augmentations.


---

# Purpose of each file

## Dataset.py

Handles dataset loading, preprocessing, augmentation, and PyTorch DataLoader creation for ConvNeXt-Tiny.

### Given Dataset Breakdown

ADNI dataset (**30,250 files**)

```text
test (9,000)
├── NC (4,540)
└── AD (4,460)

train (21,520)
├── NC (11,120)
└── AD (10,400)
```

- Total NC = **15,660 files**
- Total AD = **14,860 files**

The ADNI dataset contains 30,520 MRI images, comprising 15,660 NC and 14,860 AD images. With a relatively balanced class distribution, the dataset can be used directly for binary classification without additional class-balancing techniques.

The original training set contains **21,520 images (70.51%)**.

Of the **9,000** images in the original test set, **30%** are allocated to validation, resulting in:

| Split | Images | Percentage |
|-------|-------:|-----------:|
| Train | 21,520 | 70.51% |
| Validation | 2,700 | 8.85% |
| Test | 6,300 | 20.64% |

### Reason for the Split

The dataset follows an approximately **70/10/20** training, validation, and test split.

### Preprocessing and Augmentation (Train)

- Convert images to grayscale.
- Crop black borders and resize to **224 × 224** pixels.
- Normalize pixel values.
- Apply random affine transformations and brightness/contrast adjustments during training to reduce overfitting.
- Load images in batches using PyTorch DataLoader, with GPU transfer handled during training.

> Validation and test images use consistent preprocessing without augmentation.


## Modules.py

Build the structure of the model.

### Model Architecture

ConvNeXt is a pure convolutional neural network (ConvNet) that modernizes the traditional ResNet architecture by incorporating design choices inspired by Vision Transformers (ViTs). Its key insight is that a conventional CNN can achieve competitive accuracy, scalability, and efficiency without using self-attention.

### A Modernized CNN

| Feature | Normal CNN | ConvNeXt |
|---------|------------|----------|
| Kernel Size | 3×3 | 7×7 depthwise |
| Activation | ReLU | GELU |
| Normalization | Batch Normalization | Layer Normalization |
| Stem | 7×7 conv + max pool | 4×4 stride-4 patchify conv |
| Block Design | Bottleneck | Inverted bottleneck |
| Down Sampling | Stride convs | Separate 2×2 stride-2 conv |

### The model for ADNI dataset

- ConvNeXt-Tiny implemented from scratch in PyTorch.
- Binary classification of single-channel ADNI brain slices.

ConvNeXt block applies:

- 7×7 depthwise convolution
- LayerNorm
- Inverted bottleneck MLP
- Layer Scale (1e-6)
- Stochastic Depth
- Residual Connection

The drop rate increases linearly across the **18 blocks**, reaching **0.3**.


## Train.py

The training configuration defines:

- epochs
- learning rate
- random seed

The model is defined and saved, with training managed by `run_epoch`.

Data loaders and model components are imported from `dataset.py` and `modules.py`.

Training uses:

- Cross-Entropy Loss with label smoothing
- AdamW optimizer
- Learning rate scheduler

Each epoch records:

- Training loss
- Validation loss
- Accuracy

The best-performing model checkpoint is saved.

Finally, loss curves, accuracy curves, classification report, confusion matrix, and NC/AD performance breakdown are printed.


## Predict.py

The trained model is loaded and evaluated on the test set.

Outputs include:

- Confusion matrix
- Correct predictions
- Incorrect predictions
- Grad-CAM visualizations highlighting image regions influencing predictions.


---

# Comparison between 2 models (ConvNeXt and Basic MLP)

## Chosen Baseline Model — Basic MLP

All other variables are kept constant to ensure a fair comparison, with model architecture being the only variable that changes.

### Controlled Variables

1. Seed = 0
2. Epoch = 30
3. Batch size = 32
4. Optimizer = Adam
5. Parameters = 27.82M
6. Learning Rate = 1e-4
7. Loss Function = Cross Entropy Loss
8. Train/val/test split (20% of train patients for validation)
9. Same preprocessing and augmentations from `dataset.py`
10. Evaluation metrics (loss, accuracy, AUROC, classification report, confusion matrix, peak VRAM)
11. Best validation checkpoint selected for testing.

### Trial and Errors

1. Chose ResNet-50 as baseline because it can be modified from Demo 2 ResNet-18.
2. Result: ResNet had higher test accuracy than ConvNeXt.
3. Confusion: unclear if all parameters need to be the same including normalization and activation.
4. Realization: ResNet-50 should not be used as the baseline.
5. A Basic MLP was chosen instead.
6. Parameters were kept the same.

### Results

| Metric | ConvNeXt | Basic MLP |
|--------|---------:|----------:|
| Best Validation Accuracy | 0.8609 | 0.6898 |
| Test Accuracy | 0.7366 | 0.6207 |
| Test AUROC | 0.8176 | 0.6384 |
| Peak VRAM | 2.68 GB | 0.58 GB |

ConvNeXt outperformed the Basic MLP in validation accuracy, test accuracy, and test AUROC.


---

# Tuning the model to reach 0.8 test accuracy

## Initial Setup

20% from train used as validation.

| Change | Test Accuracy |
|--------|--------------:|
| Adam → AdamW | 0.7524 |
| Drop path = 0.1 | 0.7276 |
| Average each patient slices | 0.7600 |
| Keep patient slices 78–107 | 0.7314 |
| Crop brain region | 0.7339 |
| Strong augmentation | 0.7156 |
| Label smoothing = 0.1 | 0.7371 |
| LR = 2e-4 | 0.7491 |
| LR = 2e-4 | 0.7594 (0.7889 per patient) |
| LR = 8e-4 | 0.7506 |
| 50 epochs | 0.7590 |
| Seed = 2 | 0.7599 |
| TTA | 0.7644 |
| Ensemble of 5 models | 0.7800 per patient |
| Remove filter | 0.7594 |

### 20/10/2026 – Tutor (Jevi)

Split validation from test instead of train.

### Setup

10% from test used as validation.

| Change | Test Accuracy |
|--------|--------------:|
| AdamW LR=6e-4 | 0.7878 |
| 30% validation | 0.7689 |
| 20% validation | 0.7760 |
| Deeper network | 0.7796 |
| Smaller network | 0.7779 |
| LR=3e-4 | 0.7765 |

### 21/10/2026 – Tutor (Zhao)

Increase learning rate to **4e-3** and train for **100 epochs**.

| Change | Test Accuracy |
|--------|--------------:|
| LR=0.004 + 40 epochs | 0.7889 |
| LR=0.004 + 100 epochs | **0.8089** |
| Remove validation | 0.7971 |
| LR=0.01 | Crash (NaN) |
| LR=0.008 | 0.7792 |


---

# Final Setup

- **Dataset:** ADNI
- **Optimizer:** AdamW
- **Learning Rate:** 4e-3
- **Epoch:** 100
- **Weight Decay:** 0.05
- **Batch Size:** 32
- **Loss Function:** Cross Entropy Loss
- **Seed:** 0


---

# Results

| Metric | Value |
|--------|------:|
| Best Test Accuracy | **0.8079** |
| Test Loss | 0.6672 |
| Test AUROC | **0.8874** |

### Confusion Matrix

| | Predicted NC | Predicted AD | Total |
|---|------------:|------------:|------:|
| True NC | 3970 | 170 | 4140 |
| True AD | 1386 | 2574 | 3960 |

### NC

- Precision: **0.7412**
- Recall: **0.9589**
- F1: **0.8361**

### AD

- Precision: **0.9380**
- Recall: **0.6500**
- F1: **0.7679**

### Averages

- Macro Precision: **0.8396**
- Macro Recall: **0.8045**
- Macro F1: **0.8020**

Although this model hits a test accuracy of **0.8**, it is biased towards NC. It only gives very few AD false alarms (170), but many missed AD cases.

When the model predicts AD, it has a high chance that it is actually AD (90%).

However, when the model predicts NC, it has a higher chance compared to AD that it is actually AD.

### AUROC

**0.8874**

- Best case: **100%**
- Threshold: **≥0.5 = AD**, **≤0.5 = NC**


---

# Fault Analysis


---

# Things needed to replicate the project

## Dataset

- ADNI
- Structured into train and test with NC and AD folders
- Saved as `.jpeg`

## Python Dependencies

```text
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
numpy>=1.20.0
```

## Hardware and Software Requirements

- NVIDIA GPU with CUDA support
- Sufficient disk space
- GPU with **6GB–8GB** memory


---

# Acknowledgement of AI