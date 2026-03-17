import torch

def iou(boxes1, boxes2):
    b1_x1 = boxes1[..., 0:1] - boxes1[..., 2:3] / 2
    b1_y1 = boxes1[..., 1:2] - boxes1[..., 3:4] / 2
    b1_x2 = boxes1[..., 0:1] + boxes1[..., 2:3] / 2
    b1_y2 = boxes1[..., 1:2] + boxes1[..., 3:4] / 2

    b2_x1 = boxes2[..., 0:1] - boxes2[..., 2:3] / 2
    b2_y1 = boxes2[..., 1:2] - boxes2[..., 3:4] / 2
    b2_x2 = boxes2[..., 0:1] + boxes2[..., 2:3] / 2
    b2_y2 = boxes2[..., 1:2] + boxes2[..., 3:4] / 2

    x1_intersect = torch.max(b1_x1, b2_x1)
    y1_intersect = torch.max(b1_y1, b2_y1)
    x2_intersect = torch.min(b1_x2, b2_x2)
    y2_intersect = torch.min(b1_y2, b2_y2)

    intersect_area = (x2_intersect - x1_intersect).clamp(0) * (y2_intersect - y1_intersect).clamp(0)

    b1_area = abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
    b2_area = abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1))

    union_area = b1_area + b2_area - intersect_area

    return intersect_area / (union_area + 1e-6)