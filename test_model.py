import torch
from src.model import TinyDetector

def test_forward_pass():
    model = TinyDetector()
    dummy_input = torch.randn(2, 3, 448, 448)

    output = model(dummy_input)
    print(f"Input format: {dummy_input.shape}")
    print(f"Output format: {output.shape}")

    assert output.shape == (2, 7, 7, 30), "Error: output shape should be (2, 7, 7, 30)"
    print("Forward pass executed and passed successfully")

if __name__ == "__main__":
    test_forward_pass()