import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.model import TinyDetector
from src.loss import YoloLoss
from src.dataset import VOCDataset


def get_transforms():
    return A.Compose([
        A.Resize(448, 448),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))


def train_fn(train_loader, model, optimizer, loss_fn, device, scaler):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    mean_breakdown = {"coord": 0, "obj": 0, "noobj": 0, "cls": 0}

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)

        with torch.amp.autocast('cuda'):
            out = model(x)
            loss, breakdown = loss_fn(out, y)

        optimizer.zero_grad()

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        scaler.step(optimizer)
        scaler.update()

        mean_loss.append(loss.item())
        for key in mean_breakdown:
            mean_breakdown[key] += breakdown[key]

        loop.set_postfix(loss=loss.item())

    for key in mean_breakdown:
        mean_breakdown[key] /= len(train_loader)

    return sum(mean_loss) / len(mean_loss), mean_breakdown


def main():
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 5e-4
    BATCH_SIZE = 16
    EPOCHS = 100
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = TinyDetector(grid_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = YoloLoss(S=7, B=2, C=20)

    scaler = torch.amp.GradScaler('cuda')

    transform = get_transforms()

    dataset = VOCDataset(data_dir="data/dummy_voc", split="train", transform=transform)

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        model.train()
        avg_loss, breakdown = train_fn(train_loader, model, optimizer, loss_fn, DEVICE, scaler)

        scheduler.step()

        print(f"Mean Loss: {avg_loss:.4f}")
        print(
            f"Breakdown -> Coord: {breakdown['coord']:.4f} | Obj: {breakdown['obj']:.4f} | NoObj: {breakdown['noobj']:.4f} | Cls: {breakdown['cls']:.4f}")

        if epoch == 1:
            print("\nTeste de sanidade concluído. Interrompendo loop fictício.")
            break


if __name__ == "__main__":
    main()