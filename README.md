# Evaluating the Trade-Offs of Explainable AI for Galaxy Classification

A PyTorch implementation comparing three prominent Explainable AI (XAI) methods — **Grad-CAM**, **LIME**, and **DeepSHAP** — applied to a fine-tuned ResNet-18 model for four-class galaxy morphology classification. The study quantitatively evaluates each method on computational efficiency and explanation fidelity, providing a framework for selecting appropriate XAI tools in scientific applications.

Full analysis is documented in the accompanying paper: *"Evaluating the Trade-Offs of Explainable AI for Galaxy Classification"*.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture and Training](#model-architecture-and-training)
- [XAI Methods](#xai-methods)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [References](#references)

---

## Project Overview

Deep learning classifiers for galaxy morphology often achieve high accuracy but remain opaque in their reasoning. This project investigates whether the regions a model uses to classify galaxy types are scientifically meaningful — such as spiral arms, central bulges, or disc orientation — by applying three XAI techniques and measuring how faithfully each one reflects the model's actual decision process.

**Key finding:** There is a clear trade-off between speed and fidelity. Grad-CAM, despite being the slowest method in this implementation, produced significantly more faithful explanations and achieved the highest Fidelity-Efficiency Score (FES), making it the recommended tool for scientific contexts where trustworthiness is critical.

---

## Repository Structure

```
Github/
├── galaxy/               # Test dataset — one subdirectory per class
│   ├── edge_on/
│   ├── elliptical/
│   ├── spiral/
│   └── other/
├── model/                
|   ├── edge_on/
│   ├── elliptical/
│   ├── spiral/
│   ├── other/
│   └── model.pt          # Saved model weights  
├── outputs/              # Generated XAI saliency maps
│   ├── gradcam/
│   │   ├── edge_on/
│   │   ├── elliptical/
│   │   ├── spiral/
│   │   └── other/
│   ├── lime/
│   │   ├── edge_on/
│   │   ├── elliptical/
│   │   ├── spiral/
│   │   └── other/
│   └── deepshap/
│       ├── edge_on/
│       ├── elliptical/
│       ├── spiral/
│       └── other/
├── main.ipynb            # Full pipeline: training, XAI generation, metrics
└── README.md
```

> **Note:** Training data (`model/`) is not included in this repository due to size constraints. The `galaxy/` directory contains the balanced test set (140 images per class, 560 total). Model weights (`model.pt`) are required to run inference and XAI generation — see [Usage](#usage).

---

## Dataset

The classifier targets four galaxy morphology classes:

| Class | Description |
|---|---|
| `edge_on` | Galaxies viewed along their disc plane |
| `elliptical` | Smooth, ellipsoidal galaxies with no disc |
| `spiral` | Galaxies with visible spiral arm structure |
| `other` | Irregular or otherwise unclassifiable morphologies |

**Training set:** ~3,000–5,000 images per class (custom-curated), split 80/20 into train and validation subsets. Class imbalance was handled using PyTorch's `WeightedRandomSampler`.

**Test set:** 140 images per class (560 total), held out and strictly balanced.

**Preprocessing pipeline:**
- Resize to 224×224 pixels
- Training only: `TrivialAugmentWide` data augmentation
- Normalize with per-channel mean `(0.4914, 0.4822, 0.4465)` and std `(0.2023, 0.1994, 0.2010)`

---

## Installation

**Requirements:**
- Python 3.8+
- PyTorch with CUDA support (T4 GPU or equivalent recommended)
- The following libraries:

```bash
pip install grad-cam lime captum
pip install torch torchvision scikit-learn scikit-image matplotlib opencv-python tqdm
```

The notebook was developed and run on **Google Colab** using a **T4 GPU**. All Drive paths in `main.ipynb` will need to be updated to match your local or cloud storage layout.

---

## Usage

All steps are contained in `main.ipynb`. Run the cells in order:

1. **Install dependencies** — installs `grad-cam`, `lime`, and `captum`
2. **Train the model** — fine-tunes ResNet-18 and saves `model.pt`
3. **Generate XAI saliency maps** — runs Grad-CAM, LIME, and DeepSHAP over the test set and saves outputs to `outputs/`
4. **Compute metrics** — measures runtime, MCD fidelity curves, FES scores, and classification performance

> Before running, update all file paths (prefixed `/content/drive/MyDrive/Paper/`) to point to your dataset and model locations.

To load the saved model for standalone inference:

```python
import torch
from torchvision import models
import torch.nn as nn

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load("model/model.pt"))
model.eval()
```

---

## Model Architecture and Training

The classifier is built on **ResNet-18** with ImageNet pretrained weights. Only the final fully connected layer was modified to output 4 classes; all other weights were retained and fine-tuned.

| Hyperparameter | Value |
|---|---|
| Architecture | ResNet-18 (pretrained on ImageNet) |
| Input size | 224 × 224 |
| Output classes | 4 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Batch size | 128 |
| Epochs | 25 |
| Loss function | Cross-Entropy |
| Class balancing | `WeightedRandomSampler` |
| Mixed precision | `torch.amp.autocast` (fp16) |
| Scheduler | `ReduceLROnPlateau` (patience=3, factor=0.5) |

---

## XAI Methods

Three XAI techniques are applied to the trained model to generate per-image saliency maps:

### 1. Grad-CAM
Implemented via the `pytorch_grad_cam` library. Targets the final convolutional block (`model.layer4[-1]`) of ResNet-18. Produces a coarse localization heatmap upsampled with `cv2.resize` and overlaid on the original image. Explanations are generated with respect to the model's own predicted class using `ClassifierOutputTarget`.

### 2. LIME
Implemented via the `lime` library (`lime_image`). Configured with **200 perturbed samples** and **top-10 features** to manage high RAM consumption observed during perturbation. Generates superpixel-based segmented explanations. This constrained configuration was chosen deliberately to test performance under resource-limited settings.

### 3. DeepSHAP (via GradientSHAP)
Implemented via Captum's `GradientShap`. In-place ReLU operations in ResNet-18 are replaced with out-of-place equivalents before running. A baseline distribution of **10 random noise tensors** is used. Attributions are computed with `stdevs=0.0001` noise smoothing. The resulting pixel-level attribution map is summed across channels, normalized, and overlaid on the original image using the `inferno` colormap.

---

## Evaluation Metrics

### Classification Performance
Standard metrics on the 560-image balanced test set: Overall Accuracy, Weighted F1-Score, Weighted Jaccard Similarity, and per-class Precision / Recall / F1.

### Computational Efficiency
Average wall-clock time (seconds) to generate a single explanation, measured across the full test set.

### Fidelity — Mean Confidence Drop (MCD)
For each threshold `t` in `[0.05, 0.10, 0.15, 0.25, 0.50]`, the top `t%` of pixels identified by the explainer are retained; the rest are masked. MCD measures the average drop in the model's confidence on the true class:

```
MCD = (1/N) × Σ [ f(x_i)_true − f(x_i,masked)_true ]
```

A steeper drop at low thresholds indicates a more faithful explanation — the explainer has correctly identified the pixels most critical to the model's decision.

### Fidelity-Efficiency Score (FES)
A composite score balancing fidelity and speed:

```
FES = AUC_MCD / T_avg
```

where `AUC_MCD` is the area under the MCD curve and `T_avg` is the average runtime per image. Higher FES indicates a better overall trade-off.

---

## Results

### Classification Performance

| Metric | Value |
|---|---|
| Overall Accuracy | 0.9893 |
| F1 Score (Weighted) | 0.9893 |
| Jaccard Similarity (Weighted) | 0.9788 |

Per-class breakdown (test set, 140 images each):

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| edge_on | 1.00 | 0.99 | 0.99 |
| elliptical | 0.99 | 0.99 | 0.99 |
| other | 0.99 | 0.99 | 0.99 |
| spiral | 0.99 | 1.00 | 0.99 |

### XAI Comparison

| Method | Avg Runtime (s/image) | FES Score |
|---|---|---|
| DeepSHAP | 0.060 | ≈ 0.95 |
| LIME | 0.069 | ≈ 0.56 |
| Grad-CAM | 0.361 | ≈ 1.23 |

**Key findings:**
- Grad-CAM was ~6× slower than DeepSHAP in this implementation, likely due to overhead in the `pytorch_grad_cam` library rather than the algorithm itself — this is counter to typical benchmarks where Grad-CAM is often the fastest method.
- LIME's speed was achieved only at the cost of explanation quality (200 samples, 10 features), and the process was still RAM-intensive, revealing a three-way trade-off between fidelity, computation time, and memory.
- Grad-CAM's MCD curve dropped by over 60% when retaining only the top 5% of pixels, far outperforming DeepSHAP and LIME (both under 10% drop at top 15%). Its superior fidelity more than compensated for slower runtime, yielding the highest FES.

### Qualitative Outputs

**Grad-CAM** — coarse heatmap highlighting the broad region deemed most important:

![Grad-CAM overlay](output/gradcam/spiral/154505.jpg)

**LIME** — superpixel segmentation identifying locally influential regions:

![LIME segmentation](output/lime/spiral/154505.jpg)

**DeepSHAP** — pixel-level attribution map showing positive and negative contributions:

![DeepSHAP overlay](output/deepshap/spiral/154505.jpg)

---

## References

1. M. Mohammadi et al., "Detection of extragalactic Ultra-compact dwarfs and Globular Clusters using Explainable AI techniques," *Astronomy and Computing*, vol. 39, 2022.
2. K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," CVPR 2016, pp. 770–778.
3. S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," NeurIPS 2017.
4. M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why Should I Trust You?': Explaining the Predictions of Any Classifier," KDD 2016, pp. 1135–1144.
5. R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," ICCV 2017, pp. 618–626.
6. J. Brasse et al., "Explainable artificial intelligence in information systems," *Electronic Markets*, vol. 33, no. 26, 2023.
7. R. C. Fong and A. Vedaldi, "Interpretable Explanations of Black Boxes by Meaningful Perturbation," ICCV 2017.
8. C. J. Lintott et al., "Galaxy Zoo: morphologies derived from visual inspection of galaxies from the SDSS," *MNRAS*, vol. 389, no. 3, pp. 1179–1187, 2008.
9. S. Dieleman, K. W. Willett, and J. Dambre, "Rotation-invariant convolutional neural networks for galaxy morphology prediction," *MNRAS*, vol. 450, no. 2, pp. 1441–1459, 2015.
10. A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," NeurIPS 2019.
11. S. Hooker et al., "A Benchmark for Interpretability Methods in Deep Neural Networks," NeurIPS 2019.
12. T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of Statistical Learning*, 2nd ed. Springer, 2009.
13. A. Shrikumar, P. Greenside, and A. Kundaje, "Learning Important Features Through Propagating Activation Differences," ICML 2017, pp. 3145–3153.