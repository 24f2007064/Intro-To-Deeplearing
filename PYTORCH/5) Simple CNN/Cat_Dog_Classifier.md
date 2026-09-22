# Cat vs Dog Image Classification with PyTorch

A hands-on **CNN-based Cat vs Dog image classification** project built with **PyTorch**.

The project focuses on understanding how **data augmentation, Dropout, Batch Normalization, batch size, epochs, and learning rate** affect training loss, training accuracy, test accuracy, and generalization.

---

## Dataset

```text
PetImages/
├── Cat/
└── Dog/
```

| Class | Images |
|---|---:|
| Cat | 12,499 |
| Dog | 12,499 |
| **Total** | **24,998** |

Labels:

```text
0 → Cat
1 → Dog
```

Images are loaded as RGB and resized to **256 × 256**.

---

## CNN Architecture

```text
Input: 3 × 256 × 256
        ↓
Conv2D: 3 → 32
ReLU
MaxPool
        ↓
Conv2D: 32 → 64
ReLU
MaxPool
        ↓
Conv2D: 64 → 128
ReLU
MaxPool
        ↓
Flatten
        ↓
Linear: 115200 → 128
ReLU
        ↓
Linear: 128 → 64
ReLU
        ↓
Linear: 64 → 1
```

The model uses:

```python
nn.BCEWithLogitsLoss()
```

and `sigmoid` during prediction.

---

## Data Pipeline

```text
Image file
   ↓
PIL / RGB
   ↓
Resize
   ↓
Data Augmentation (training only)
   ↓
ToTensor
   ↓
Dataset
   ↓
DataLoader
   ↓
CNN
```

Training augmentation included experiments with:

- Random Rotation
- Random Horizontal Flip
- Random Affine
- Translation
- Scaling / Zoom
- Shearing

The test set was kept free of random augmentation.

---

# 🧪 Experiments

## 1. Previous Results (without Dropout and BatchNorm)

| Rotation | Epochs | Final Loss | Train Accuracy | Test Accuracy |
|----------|-------:|-----------:|---------------:|--------------:|
| 10° | 5 | 0.3614 | 85.25% | ~84.64% |
| 10° | 10 | 0.2736 | 89.22% | 86.48% |
| 90° | 10 | 0.3987 | 81.16% | 81.94% |

**Observation:** Increasing the rotation range made the training task harder and reduced performance on the normal, unrotated test set.

---

## 2. Results with Dropout and BatchNorm

Configuration:

```text
Dropout = 0.1
BatchNorm = Yes
Rotation = 90°
```

| Dropout | BatchNorm | Rotation | Epochs | Final Loss | Train Accuracy | Test Accuracy |
|---------|-----------|----------|-------:|-----------:|---------------:|--------------:|
| 0.1 | Yes | 90° | 10 | 0.4892 | 78.32% | 79.64% |
| 0.1 | Yes | 90° | 15 | 0.4382 | 80.41% | 82.52% |

**Observation:** Increasing training from 10 to 15 epochs improved the measured test accuracy from 79.64% to 82.52%.

---

## 3. Random Affine / Stronger Data Augmentation

Configuration:

```text
Dropout = 0.1
BatchNorm = Yes
Rotation = 90°
Random Affine = Yes
```

| Dropout | BatchNorm | Rotation | Epochs | Final Loss | Train Accuracy | Test Accuracy | Batch Size |
|---------|-----------|----------|-------:|-----------:|---------------:|--------------:|-----------:|
| 0.1 | Yes | 90° | 15 | 0.4791 | 75.69% | 79.72% | 32 |
| 0.1 | Yes | 90° | 20 | 0.415782 | 82.66% | 85.28% | 32 |
| 0.1 | Yes | 90° | 20 | 0.4550 | 79.47% | 83.64% | 64 |

**Observation:** The 20-epoch, batch-size-32 experiment reached the highest measured test accuracy in this project: **85.28%**.

For the same general augmented configuration, batch size 32 produced a higher measured test accuracy than batch size 64. This is an observation from this experiment, not a universal rule.

---

#  Learning Rate Experiment

The main optimizer used was **Adam**.

An Adam learning rate of `0.01` was tested:

```text
Epoch 1 → 0.8231
Epoch 2 → 0.6935
Epoch 3 → 0.7001
Epoch 4 → 0.6933
Epoch 5 → 0.6934
```

The model reached about **50% training accuracy** and produced nearly identical outputs for different images.

Using:

```python
lr = 0.001
```

produced much more stable learning.

This demonstrates that learning rate should be chosen together with the optimizer and model configuration. `0.01` is not inherently bad, but it was too aggressive for this particular CNN + Adam setup.

---

Epochs explored:

```text
10, 15, 20
```

Batch sizes explored:

```text
32, 64, 128
```
