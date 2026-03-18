import torch
from src.nms import non_max_suppression

def test_nms():
    predictions = torch.tensor([
        [0.0, 0.9, 0.5, 0.5, 0.2, 0.2],
        [0.0, 0.8, 0.52, 0.52, 0.2, 0.2],
        [0.0, 0.7, 0.1, 0.1, 0.1, 0.1],
        [1.0, 0.85, 0.5, 0.5, 0.2, 0.2],
        [0.0, 0.2, 0.8, 0.8, 0.1, 0.1]
    ])

    resultado = non_max_suppression(
        predictions,
        iou_threshold=0.5,
        prob_threshold=0.4
    )

    assert len(resultado) == 3, f"Error: Expected 3 bboxes, got {len(resultado)}"

    scores = [box[1] for box in resultado]
    assert 0.2 not in scores, 'Error: NMS not filtered bbox with low scores'

    print("NMS test executed successfully")

if __name__ == "__main__":
    test_nms()