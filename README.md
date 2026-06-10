---
layout: default
title: Homework 4
permalink: /homework/04/index.html
---

# Diego's Lab Report: Driving with Transformers and CNNs

**Course:** Deep Learning (Master's Degree Program)  
**Institution:** The University of Texas at Austin (UT Austin)  
---

## Overview
In this homework, I implemented and evaluated three different neural network architectures—an MLP, a Transformer, and a CNN—to act as trajectory planners in the PySuperTuxKart driving environment. The planners predict future vehicle waypoints based on either ground-truth track boundaries (perfect vision) or direct camera images (end-to-end vision).

This document serves as a summary and lab report of the actions I took, the architectures I designed, and the training strategy I implemented to successfully complete this project.

---

## 1. Data Preprocessing & Augmentation

To train robust driving models, I modified the dataset loading and transform pipeline:
* **Custom State Flipping (`StateFlip`):** In [road_transforms.py](file:///home/diegof/Documents/master_degree_repos/deeplearning/homework4-deeplearning/homework/datasets/road_transforms.py), I implemented a spatial augmentation that randomly mirrors the driving state horizontally (probability $p = 0.5$). When triggered, it swaps the left and right lane boundaries, negates the lateral coordinates for boundaries and target waypoints, and mirrors the camera image.
* **Color Jittering (`ColorJitter`):** Applied random brightness, contrast, saturation, and hue changes to the images to make the CNN planner more robust to lighting variations.
* **Transform Pipelines:** In [road_dataset.py](file:///home/diegof/Documents/master_degree_repos/deeplearning/homework4-deeplearning/homework/datasets/road_dataset.py), I configured:
  * `"aug"`: Active during training; integrates `ImageLoader`, `EgoTrackProcessor`, `StateFlip`, and `ColorJitter`.
  * `"state_only"`: Active for MLP/Transformer validation; returns coordinates without loading raw images.
  * `"default"`: Active for CNN validation; returns un-augmented camera images and coordinates.

---

## 2. Model Architectures & Design Decisions

All models were implemented in [models.py](file:///home/diegof/Documents/master_degree_repos/deeplearning/homework4-deeplearning/homework/models.py):

### A. MLP Planner (`MLPPlanner`)
* **Objective:** Map track boundaries directly to waypoints.
* **Input:** Left and right boundary points `(B, n_track, 2)` each.
* **Target Passing Criteria:** Longitudinal error < **0.2**, Lateral error < **0.6**.
* **My Architecture:**
  * **Input Scaling:** Divided raw coordinate coordinates by $15.0$ to scale them into a stable range.
  * **Flattening:** Concatenated left and right boundaries into a single feature vector of size $4 \times n_{\text{track}} = 40$.
  * **Network Structure:** A linear projection to 512 dimensions with Batch Normalization and ReLU, followed by two residual layers (each with 512 hidden units, BatchNorm, and ReLU). 
  * **Residual Connections:** Added the input of each intermediate block to its output (`x = x + layer(x)`) to improve gradient flow during training.
  * **Output Projection:** Projected the final representations to $2 \times n_{\text{waypoints}} = 6$ outputs, reshaped to `(B, n_waypoints, 2)`.

### B. Transformer Planner (`TransformerPlanner`)
* **Objective:** Utilize cross-attention to query driving track boundaries.
* **Target Passing Criteria:** Longitudinal error < **0.2**, Lateral error < **0.6**.
* **My Architecture:**
  * **Input Projection:** Concatenated left and right boundaries `(B, 20, 2)` and mapped them to `d_model = 64` dimensions via a linear projection.
  * **Positional Embeddings:** Added learned 1D spatial embeddings to each of the 20 boundary points to retain sequence order information.
  * **Waypoint Querying:** Defined learned query embeddings of shape `(n_waypoints, d_model)`.
  * **Cross-Attention:** Passed the queries and boundary key/value memory to a `nn.TransformerDecoder` (3 layers, 4 attention heads, feedforward dimension of 256). The queries attend to the road boundary representations.
  * **Output Projection:** Projected the decoder output tokens back to 2D coordinates.

### C. Vision CNN Planner (`CNNPlanner`)
* **Objective:** Direct end-to-end waypoint prediction from camera pixels.
* **Target Passing Criteria:** Longitudinal error < **0.30**, Lateral error < **0.45**.
* **My Architecture:**
  * **Preprocessing:** Standardized input images using pre-computed channel means and standard deviations.
  * **Backbone:** Initiated with a $3 \times 3$ stride-2 convolution mapping 3 channels to 32. Followed this with three residual downsampling blocks doubling channel depth at each step:
    1. Res Block 1: $32 \to 64$ channels, stride 2.
    2. Res Block 2: $64 \to 128$ channels, stride 2.
    3. Res Block 3: $128 \to 256$ channels, stride 2.
    * Each residual block uses a $1 \times 1$ conv projection shortcut in the skip path to align dimensions and channels.
  * **Planning Head:** Used `nn.AdaptiveAvgPool2d(1)` to obtain a global image representation, flattened it, and passed it through a two-layer MLP ($256 \to 128 \to 6$ outputs).

---

## 3. Training & Optimization Strategy

The training pipeline was implemented in [train_planner.py](file:///home/diegof/Documents/master_degree_repos/deeplearning/homework4-deeplearning/homework/train_planner.py):
* **Loss Function:** Used Mean Absolute Error (L1 Loss) as a robust regression objective. Importantly, the loss is computed **only on valid waypoints** by applying the boolean `waypoints_mask`.
* **Optimizer & Scheduler:**
  * Used `AdamW` optimizer (learning rate $10^{-3}$, weight decay $10^{-4}$) to stabilize training.
  * Employed a `ReduceLROnPlateau` scheduler (factor 0.5, patience 5) to dynamically decrease learning rate when validation loss stagnates.
* **Logging:** Integrated PyTorch's TensorBoard `SummaryWriter` to record training and validation L1 errors per epoch.
* **Weights Export:** Serialized model state dicts to the root directory for grading (`mlp_planner.th`, `transformer_planner.th`, `cnn_planner.th`) and archived copies in timestamped run directories.

---

## 4. Evaluation and Validation

I validated the correctness of the implementation using the provided grading test suite in [tests.py](file:///home/diegof/Documents/master_degree_repos/deeplearning/homework4-deeplearning/grader/tests.py).
* **Metric Formulation:** Evaluated performance using Longitudinal (forward direction) and Lateral (steering direction) errors.
* **Submission Zip:** Verified the final artifact using the `bundle.py` script to package code and trained weights under 60MB.
* **Running headless simulator:** Handled/documented the known headless segmentation faults with `PySuperTuxKart` during validation by isolating model weight training and testing processes.

