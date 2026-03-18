import torch
from src.evaluate import mean_average_precision

def test_map():
    pred_boxes = [
        [0, 0, 0.9, 0.5, 0.5, 0.2, 0.2], # Perfect TP
        [0, 0, 0.8, 0.1, 0.1, 0.1, 0.1], # FP
        [1, 1, 0.9, 0.5, 0.5, 0.2, 0.2], # Perfect TP
    ]

    true_boxes = [
        [0, 0, 1.0, 0.5, 0.5, 0.2, 0.2], # Image 0 target
        [1, 1, 1.0, 0.5, 0.5, 0.2, 0.2], # Image 1 target 1
        [1, 1, 1.0, 0.8, 0.8, 0.1, 0.1], # Image 1 target 2
    ]

    map_val = mean_average_precision(
        pred_boxes,
        true_boxes,
        iou_threshold=0.5,
        num_classes=2
    )

    assert map_val > 0, 'Error: mAP wrong calculation'
    print(f"mAP test executed successfully. Simulated value: {map_val.item():4f}")

if __name__ == '__main__':
    test_map()