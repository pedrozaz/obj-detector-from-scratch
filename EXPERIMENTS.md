# Experiment Decisions Log

## 1. Baseline (From-Scratch training)
- **Goal**: Train a simplified YOLO architecture from scratch with random weights on Pascal VOC 2012.
- **Result**: mAP@0.5 of 0.0049.
- **Failure Reason**: Reducing the DetectionHead parameters to
fit the 6GB VRAM constraint severely limited spatial abstraction. Combined
with the lack of pre-training, the model overfitted to predicting background in most cells, resulting in excessive false negatives.

## 2. Focal Loss
- **Goal**: Implement Focal Loss to balance the massive background prediction.
- **Result**: mAP@0.5 of 0.0000.
- **Failure Reason**: The modulation factor zeroed the gradients for background predictions
early on (which were already close zero). The network stopped penalizing the background, flooding the grid with false positives and collapsing precision.

## Decision: Transfer Learning (ResNet18)
- Since object detection datasets (like VOC) lack the volume needed to teach basic feature extraction (edges, textures)
to networks initialized from scratch, the custom backbone is discarded.
- A ResNet18 pre-trained on ImageNet will be adopted. The backbone will be frozen to ensure advanced feature extraction, and only the DetectionHead will be trained, enabling convergence within the hardware's physical memory limits.

## 3. Transfer Learning (Frozen ResNet18)
- **Goal**: Replace custom backbone with pre-trained ResNet18 (frozen) to improve feature extraction.
- **Result**: mAP@0.5 of 0.0356.
- **Decision**: The mAP improved by ~7x but remains unviable. The frozen backbone prevents spatial regression
adaptation. The next step is full Fine-Tuning: unfreezing the backbone and training the entire network with a lower learning-rate (1e-5).