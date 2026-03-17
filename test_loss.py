import torch
from src.loss import YoloLoss

def test_yolo_loss():
    predictions = torch.randn(2, 7, 7, 30, requires_grad=True)
    target = torch.zeros(2, 7, 7, 30)

    target[0, 3, 3, 20:24] = torch.tensor([0.5, 0.5, 0.3, 0.3])
    target[0, 3, 3, 24] = 1.0
    target[0, 3, 3, 5] = 1.0

    loss_fn = YoloLoss(S=7, B=2, C=20)
    loss = loss_fn(predictions, target)

    loss.backward()

    assert not torch.isnan(loss), "Error: loss resulted in NaN"
    assert loss.item() >= 0, "Error: loss cannot be negative"
    assert predictions.grad is not None, "Error: gradients are not propagated"

    print(f'Calculus executed successfully. Loss value: {loss.item():.4f}')

if __name__ == '__main__':
    test_yolo_loss()