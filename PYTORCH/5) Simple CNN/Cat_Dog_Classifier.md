# 🐱🐶 Cat vs Dog Image Classification with PyTorch

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

# 📉 Learning Rate Experiment

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

## Training Configuration

Typical settings:

```text
Optimizer     : Adam
Learning Rate : 0.001
Image Size    : 256 × 256
Loss          : BCEWithLogitsLoss
Device        : CUDA when available
GPU           : NVIDIA RTX 3060
```

Epochs explored:

```text
5, 10, 15, 20
```

Batch sizes explored:

```text
32, 64, 128
```

---

## CUDA

CUDA is used when available:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)
```

Training was performed on an **NVIDIA RTX 3060**.

---

## Predicting a New Image

A new image can be passed through the same preprocessing used for the test set:

```text
New image
   ↓
Read image
   ↓
RGB
   ↓
Resize
   ↓
ToTensor
   ↓
Add batch dimension
   ↓
CNN
   ↓
Sigmoid
   ↓
Cat / Dog
```

Example:

```python
image = Image.open("new_image.jpg").convert("RGB")
image = test_transform(image)
image = image.unsqueeze(0)

model.eval()

with torch.no_grad():
    output = model(image.to(device))
    probability = torch.sigmoid(output)
    prediction = (probability >= 0.5).long().item()

print("Dog" if prediction == 1 else "Cat")
```

---

# 📊 Key Findings

### Data augmentation

Stronger augmentation makes training harder because the model sees a wider variety of images.

### Dropout

Dropout reduced training accuracy, acting as a regularization mechanism.

### Batch Normalization

BatchNorm changed the training dynamics and, together with Dropout, produced a smaller train-test gap in the experiments.

### More epochs

Longer training helped the augmented model. In particular:

```text
15 epochs → 79.72% test accuracy
20 epochs → 85.28% test accuracy
```

for the recorded batch-size-32 augmented setup.

### Batch size

In the recorded 20-epoch augmented experiment:

```text
Batch 32 → 85.28%
Batch 64 → 83.64%
```

---

## ⚠️ Important Experimental Note

The normal test set was not randomly augmented.

Therefore, these test accuracies measure performance on **normal unseen images**. They do not directly measure robustness to rotated or otherwise transformed images.

A separate test set with controlled transformations would be needed to evaluate augmentation-specific robustness.

---

# 🏆 Best Measured Result

The highest measured test accuracy in the experiments recorded here was:

## **85.28%**

Configuration:

```text
Dropout       : 0.1
BatchNorm     : Yes
Rotation      : 90°
Random Affine : Yes
Epochs        : 20
Batch Size    : 32
```

This is the best result **among the configurations tested so far**.

---

## 🛠️ Technologies

- Python
- PyTorch
- TorchVision
- NumPy
- PIL
- Matplotlib
- CUDA

---

## 📁 Project Structure

```text
Cat-Dog-CNN/
│
├── dataset/
│   └── PetImages/
│       ├── Cat/
│       └── Dog/
│
├── notebooks/
│   └── cat_dog_cnn.ipynb
│
├── models/
│   └── model.pth
│
├── README.md
└── requirements.txt
```

---

## 🚀 Future Experiments

- Compare 10°, 20°, 30°, and 90° rotation
- Test the models on rotated test images
- Compare augmentation techniques individually
- Explore learning-rate scheduling
- Compare ReLU and LeakyReLU
- Measure images/second and epoch time
- Compare the custom CNN with transfer learning models such as ResNet or MobileNet
