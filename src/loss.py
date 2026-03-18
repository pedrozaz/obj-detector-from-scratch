import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 0.5

    def forward(self, predictions, target):
        box1 = predictions[..., self.C:self.C+4]
        box2 = predictions[..., self.C+5:self.C+9]

        target_box = target[..., self.C:self.C+4]

        iou_b1 = iou(box1, target_box)
        iou_b2 = iou(box2, target_box)

        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, bestbox = torch.max(ious, dim=0)

        obj = target[..., self.C+4].unsqueeze(3)
        noobj = 1 - obj

        # Coords selection
        best_box_coords = (1 - bestbox) * box1 + bestbox * box2

        box_predictions = best_box_coords.clone()
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )

        box_targets = target_box.clone()
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions * obj, end_dim=-2),
            torch.flatten(box_targets * obj, end_dim=-2)
        )

        # Objectness loss
        pred_box_conf = (1 - bestbox) * predictions[..., self.C+4:self.C+5] + bestbox * predictions[..., self.C+9:self.C+10]

        object_loss = self.mse(
            torch.flatten(obj * pred_box_conf),
            torch.flatten(obj * iou_maxes)
        )

        # No-Object loss
        no_object_loss = self.mse(
            torch.flatten(noobj * predictions[..., self.C+4:self.C+5], start_dim=1),
            torch.flatten(noobj * target[..., self.C+4:self.C+5], start_dim=1)
        )
        no_object_loss += self.mse(
            torch.flatten(noobj * predictions[..., self.C+9:self.C+10], start_dim=1),
            torch.flatten(noobj * target[..., self.C+4:self.C+5], start_dim=1)
        )

        # Class loss
        class_loss = self.mse(
            torch.flatten(obj * predictions[..., :self.C], end_dim=-2),
            torch.flatten(obj * target[..., :self.C], end_dim=-2)
        )

        # Total loss
        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        # Breakdown / Logs
        loss_breakdown = {
            'coord': (self.lambda_coord * box_loss).item(),
            'obj': object_loss.item(),
            'noobj': (self.lambda_noobj * no_object_loss).item(),
            'cls': class_loss.item()
        }

        return loss, loss_breakdown

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
