import torch
from src.loss import iou

def non_max_suppression(predictions, iou_threshold=0.5, prob_threshold=0.4):
    mask = predictions[:, 1] > prob_threshold
    predictions = predictions[mask]

    sort_indices = torch.argsort(predictions[:, 1], descending=True)
    predictions = predictions[sort_indices]

    bboxes_after_nms = []

    while len(predictions) > 0:
        chosen_box = predictions[0]
        bboxes_after_nms.append(chosen_box.tolist())

        if len(predictions) == 1:
            break

        chosen_box_tensor = chosen_box[2:6].unsqueeze(0)
        rest_boxes_tensor = predictions[1:, 2:6]

        ious = iou(chosen_box_tensor, rest_boxes_tensor).view(-1)

        same_class_mask = predictions[1:, 0] == chosen_box[0]
        iou_mask = ious > iou_threshold

        keep_mask = ~(same_class_mask & iou_mask)

        predictions = predictions[1:][keep_mask]

    return bboxes_after_nms