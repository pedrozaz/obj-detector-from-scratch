import torch
from collections import Counter
from src.loss import iou
from src.nms import non_max_suppression

def get_boxes(loader, model, iou_threshold, threshold, device='cuda'):
    all_pred_boxes = []
    all_true_boxes = []
    model.eval()
    train_idx = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            predictions = model(x)

            batch_size = x.shape[0]
            for idx in range(batch_size):
                bboxes = []
                for i in range(7):
                    for j in range(7):
                        conf1 = predictions[idx, i, j, 24].item()
                        conf2 = predictions[idx, i, j, 29].item()

                        class_scores = predictions[idx, i, j, :20]
                        class_pred = class_scores.argmax().item()
                        class_prob = class_scores.max().item()

                        if conf1 > conf2:
                            box = predictions[idx, i, j, 20:24].tolist()
                            conf = conf1
                        else:
                            box = predictions[idx, i, j, 25:29].tolist()
                            conf = conf2

                        score = conf * class_prob
                        bboxes.append([class_pred, score, box[0], box[1], box[2], box[3]])

                nms_boxes = non_max_suppression(
                    torch.tensor(bboxes),
                    iou_threshold=iou_threshold,
                    prob_threshold=threshold
                )

                for nms_box in nms_boxes:
                    all_pred_boxes.append([train_idx] + nms_box)

                for i in range(7):
                    for j in range(7):
                        if y[idx, i, j, 24] == 1:
                            box = y[idx, i, j, 20:24].tolist()
                            class_true = y[idx, i, j, :20].argmax().item()
                            all_true_boxes.append([train_idx, class_true, 1.0, box[0], box[1], box[2], box[3]])

                train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes

def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, num_classes=20):
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = [box for box in pred_boxes if box[1] == c]
        ground_truths = [box for box in true_boxes if box[1] == c]

        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes > 0:
            continue

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            best_iou = 0
            best_gt_idx = -1

            for idx, gt in enumerate(ground_truth_img):
                iou_val = iou(
                    torch.tensor(detection[3:]).unsqueeze(0),
                    torch.tensor(gt[3:]).unsqueeze(0)
                ).item()

                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))

        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)