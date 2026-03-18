import os
import torch
import numpy as np
from PIL import Image
from src.dataset import VOCDataset

def setup_dummy_voc():
    base_dir = 'data/dummy_voc'
    os.makedirs(os.path.join(base_dir, 'ImageSets/Main'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'Annotations'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'JPEGImages'), exist_ok=True)

    with open(os.path.join(base_dir, 'ImageSets/Main/train.txt'), 'w') as f:
        f.write('000001\n')

    img_array = np.zeros((448, 448, 3), dtype=np.uint8)
    Image.fromarray(img_array).save(os.path.join(base_dir, 'JPEGImages/000001.jpg'))

    xml_content = "<annotation><object><name>dog</name><bndbox><xmin>101</xmin><ymin>101</ymin><xmax>301</xmax><ymax>301</ymax></bndbox></object></annotation>"
    with open(os.path.join(base_dir, "Annotations/000001.xml"), "w") as f:
        f.write(xml_content)

    return base_dir

def test_voc_dataset():
    data_dir = setup_dummy_voc()

    dataset = VOCDataset(data_dir=data_dir, split='train')
    image, target = dataset[0]

    assert target.shape == (7, 7, 30), f'Incorrect shape: {target.shape}'

    assert target[3, 3, 24] == 1.0, 'Error: Objectness score missing at right split'
    assert target[3, 3, 11] == 1.0, 'Error: One-hot encoding of class "dog" is incorrect'
    assert target[3, 3, 20] > 0, 'Error: coordinate x_center was not filled'

    print('VOCDataset test passed and grid was codified successfully')

if __name__ == '__main__':
    test_voc_dataset()