import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)

class VOCDataset(Dataset):
    def __init__(self, data_dir, split='train', S=7, B=2, C=20, transform=None):
        self.data_dir = data_dir
        self.split = split
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(VOC_CLASSES)}

        split_file = os.path.join(data_dir, 'ImageSets', 'Main', f'{split}.txt')
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.data_dir, 'JPEGImages', f'{img_id}.jpg')
        xml_path = os.path.join(self.data_dir, 'Annotations', f'{img_id}.xml')

        image = np.array(Image.open(img_path).convert('RGB'))
        bboxes = self._parse_xml(xml_path)

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations['image']
            bboxes = augmentations['bboxes']

        if isinstance(image, torch.Tensor):
            img_h, img_w = image.shape[1], image.shape[2]
        else:
            img_h, img_w = image.shape[0], image.shape[1]

        target_matrix = self._encode_target(bboxes, img_w, img_h)

        return image, target_matrix

    def _parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []

        for obj in root.findall("object"):
            name_node = obj.find("name")
            if name_node is None or name_node.text is None:
                continue

            cls_name = name_node.text.lower().strip()
            if cls_name not in self.class_to_idx:
                continue

            class_idx = self.class_to_idx[cls_name]

            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text) - 1
            ymin = float(bndbox.find("ymin").text) - 1
            xmax = float(bndbox.find("xmax").text) - 1
            ymax = float(bndbox.find("ymax").text) - 1

            bboxes.append([xmin, ymin, xmax, ymax, class_idx])

        return bboxes

    def _encode_target(self, bboxes, img_w, img_h):
        target_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        for box in bboxes:
            xmin, ymin, xmax, ymax, class_idx = box
            class_idx = int(class_idx)

            x_center = ((xmin + xmax) / 2) / img_w
            y_center = ((ymin + ymax) / 2) / img_h
            width = (xmax - xmin) / img_w
            height = (ymax - ymin) / img_h

            i, j = int(self.S * y_center), int(self.S * x_center)

            i = min(i, self.S - 1)
            j = min(j, self.S - 1)

            if target_matrix[i, j, self.C + 4] == 0:
                target_matrix[i, j, self.C + 4] = 1.0
                target_matrix[i, j, self.C:self.C+4] = torch.tensor([x_center, y_center, width, height])
                target_matrix[i, j, class_idx] = 1.0

        return target_matrix