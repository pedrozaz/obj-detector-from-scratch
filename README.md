# YOLOv1 from Scratch: Methodological Foundation

## Overview
This repository serves as a scientific and practical baseline for understanding single-stage detection architecture. It contains
a PyTorch implementation of the original YOLO algorithm built from scratch. The primary goal is to map the mathematical, structural,
and hardware bottlenecks of legacy detection systems.

The empirical findings documented here act as the theoretical groundwork for subsequent [project](https://github.com/pedrozaz/estaciona-ai] project), transitioning from this baseline to
the state-of-the-art YOLOv8 architecture.

## Hardware Constraints & Engineering
Development and training were conducted under a strict 6GB VRAM constraint (RTX 2060). This physical limitation drove key
architectural engineering decisions:
* **Gradient Accumulation**: Implemented to maintain an effective batch size of 16 using micro-batches, ensuring mathematical stability
for the optimizer without triggering Out-Of-Memory (OOM) errors.
* **Detection Head Optimization**: The dense layers were structurally reduced (from ~250M to ~25M parameters) to fit the memory
footprint during the backward pass.

## Experimental Journey
A series of controlled experiments were executed on the Pascal VOC 2012 dataset to validate convergence hypotheses. Full details
are available in `EXPERIMENTS.md` at the root.
1. **Baseline (From Scratch)**: Proved that random weight initialization without ImageNet pre-training fails to learn robust
spatial features on small datasets, leading to massive false negatives (predicting background).
2. **Focal Loss**: Attempted to mathematically penalize the background dominance. Resulted in gradient vanishing for background
classes and flooded the grid with false positives.
3. **Transfer Learning (Frozen ResNet18)**: Replaced the custom backbone with a pre-trained ResNet18, Yielded a ~7x improvement in mAP,
proving the necessity of feature extraction pre-training.
4. **Full Fine-Tuning**: Unfreezing the backbone with a micro-batch size of 2 corrupted the Batch Normalization layers, degrading
the pre-trained weights through gradient noise.

## Conclusion
The failures documented in this repository empirically validate the architectural advancements of modern detectors. The [project](https://github.com/pedrozaz/estaciona-ai)
will utilize YOLOv8 to solve these exact bottlenecks through:
* **Decoupled Heads**: Preventing spatial variance from corrupting class probabilities.
* **Mosaic Augmentation**: Simulating larger batch sizes to stabilize gradients on low-VRAM hardware, such as Raspberry PI.
* **Anchor-Free / Task-Aligned Assignment**: Eliminating the rigid 7x7 grid limitations that caused the background overfitting observed
in Experiment 1.
* **CIoU Loss**: Providing better bounding box regression than standard MSE.

## Installation & Usage
This project uses [uv](https://astral.sh/uv) for fast dependency management.

```bash
# Install dependencies
uv sync

# Run training
uv run python -m src.train

# Evaluate on validation-set
uv run python eval.py
```

