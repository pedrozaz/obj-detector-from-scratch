import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from src.model import TinyDetector
from src.nms import non_max_suppression
from src.train import get_transforms

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor']

def plot_image(image, boxes):
    im = np.array(image)
    height, width, _ = im.shape
    fig, ax = plt.subplots(1)
    ax.imshow(im)

    for box in boxes:
        class_pred = int(box[0])
        prob = box[1]
        box = box[2:]

        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2

        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            f'{CLASSES[class_pred]}: {prob:.2f}',
            color='white',
            backgroundcolor='red',
            fontsize=8
        )

    plt.axis('off')
    plt.savefig('output_prediction.png', bbox_inches='tight', dpi=300)
    print('Prediction saved to "output_prediction.png"')

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    img_dir = 'data/data/VOCdevkit/VOC2012/JPEGImages'
    sample_img = os.listdir(img_dir)[0]
    image_path = os.path.join(img_dir, sample_img)

    model = TinyDetector(grid_size=7, num_boxes=2, num_classes=len(CLASSES)).to(DEVICE)
    model.load_state_dict(torch.load(
        'checkpoints/yolo_resnet_voc2012.pth', map_location=DEVICE, weights_only=True))

    transform = get_transforms()
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transformed =transform(image=image)
    x = transformed['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        predictions = model(x)

    bboxes = []
    for i in range(7):
        for j in range(7):
            conf1 = predictions[0, i, j, 24].item()
            conf2 = predictions[0, i, j, 29].item()

            class_scores = predictions[0, i, j, :20]
            class_pred = class_scores.argmax().item()
            class_prob = class_scores.max().item()

            if conf1 > conf2:
                box = predictions[0, i, j, 20:24].tolist()
                conf = conf1
            else:
                box = predictions[0, i, j, 25:29].tolist()
                conf = conf2

            cell_x, cell_y, width, height = box
            global_center_x = (i + cell_x) / 7.0
            global_center_y = (i + cell_y) / 7.0


            score = conf * class_prob
            bboxes.append([class_pred, score, global_center_x, global_center_y, width, height])

    num_boxes = non_max_suppression(
        torch.tensor(bboxes),
        iou_threshold=0.5,
        prob_threshold=0.1
    )

    plot_image(image, num_boxes)

if __name__ == '__main__':
    main()