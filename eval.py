import torch
from torch.utils.data import DataLoader
from src.model import TinyDetector
from src.dataset import VOCDataset
from src.evaluate import get_boxes, mean_average_precision
from src.train import get_transforms

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 8
    NUM_WORKERS = 4

    model = TinyDetector(grid_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    model.load_state_dict(torch.load("checkpoints/yolo_focal_loss_voc2012.pth", map_location=DEVICE, weights_only=True))
    model.eval()

    transform = get_transforms()
    dataset = VOCDataset(data_dir="data/data/VOCdevkit/VOC2012", split="val", transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )

    print('Extracting predictions and targets...')
    pred_boxes, true_boxes = get_boxes(
        loader,
        model,
        iou_threshold=0.5,
        threshold=0.4,
        device=DEVICE
    )

    print('Calculating mAP@0.5...')
    map_val = mean_average_precision(
        pred_boxes,
        true_boxes,
        iou_threshold=0.5,
        num_classes=20
    )

    print(f'Final mAP@0.5: {map_val.item():.4f}')

if __name__ == '__main__':
    main()