import os
import torch
from src.dataset import VOCDataset


def run_diagnostics():
    dataset = VOCDataset(data_dir="data/dummy_voc", split="train")
    image, target = dataset[0]

    print("--- Diagnóstico do VOCDataset ---")
    print(f"Shape da imagem lida: {image.shape}")

    xml_path = "data/dummy_voc/Annotations/000001.xml"
    bboxes = dataset._parse_xml(xml_path)
    print(f"Bboxes extraídas do XML: {bboxes}")

    if len(bboxes) == 0:
        print("ERRO: Nenhuma bbox foi lida do XML.")
        return

    for box in bboxes:
        xmin, ymin, xmax, ymax, class_idx = box
        img_h, img_w = image.shape[0], image.shape[1]
        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h

        i, j = int(7 * y_center), int(7 * x_center)
        print(f"Centro normalizado: x_c={x_center:.4f}, y_c={y_center:.4f}")
        print(f"Célula mapeada (i, j): ({i}, {j})")

    indices = torch.nonzero(target[..., 24])
    print(f"Células com objectness == 1.0 no tensor (i, j): {indices.tolist()}")


if __name__ == "__main__":
    run_diagnostics()