# Mathematical Foundations
 
This document outlines the core mathematical formulations governing the YOLOv1 architecture implemented in this repository, alongside the theoretical basis for the failed Focal Loss experiment.
 
## 1. Multi-Part Mean Squared Error (MSE) Loss
 
The YOLOv1 loss function is a unified sum of squared errors, balanced by two constants: $\lambda_{coord} = 5$ to emphasize bounding box accuracy, and $\lambda_{noobj} = 0.5$ to penalize the overwhelming number of background cells.
 
### Bounding Box Regression Loss
 
Calculates the error for coordinates $(x, y)$ and dimensions $(w, h)$. The square root of the width and height is used to ensure that small deviations in small boxes are penalized more severely than the same deviations in large boxes.
 
$$Loss_{coord} = \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 + (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right]$$
 
### Objectness Confidence Loss
 
Evaluates the confidence score $C$. $\mathbb{1}\_{ij}^{obj}$ denotes if an object is present in cell $i$ and bounding box predictor $j$, while $\mathbb{1}_{ij}^{noobj}$ denotes the background.

$$Loss_{conf} = \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj}(C_i - \hat{C}_i)^2 + \lambda_{noobj} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{noobj}(C_i - \hat{C}_i)^2$$
 
### Classification Loss
 
Computes the squared error for the conditional class probabilities $p_i(c)$, applied only to cells containing an object.
 
$$Loss_{class} = \sum_{i=0}^{S^2} \mathbb{1}_{i}^{obj} \sum_{c \in classes} (p_i(c) - \hat{p}_i(c))^2$$
 
## 2. Focal Loss Experiment (Failed Hypothesis)
 
In Experiment 2, the standard MSE for the background was replaced with a Focal Loss modulation to handle the extreme class imbalance (grid cells vs. actual objects).
 
### Mathematical Formulation
 
A modulating factor $|\hat{C}_i - C_i|^\gamma$ (with $\gamma = 2$) was applied to the background confidence loss. Since the target $C_i$ for background is $0$, the formula simplifies to:
 
$$Loss_{noobj\text{-}focal} = \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{noobj} \cdot |\hat{C}_i|^{\gamma} \cdot \hat{C}_i^{2}$$
 
### Cause of Collapse
 
When the network initializes, predictions $\hat{C}_i$ are typically close to zero. The focal factor $|\hat{C}_i|^2$ exponentially shrank the already small gradients produced by the MSE. The network effectively stopped updating weights to suppress background predictions, resulting in a grid flooded with false positives and a $mAP@0.5 = 0.0000$.
 
 
## References
 
1. Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640). *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
2. Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002). *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*.
