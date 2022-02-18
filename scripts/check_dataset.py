"""
Check datasets
"""
import os, sys
import numpy as np
import cv2
from pathlib import Path
import random
import argparse

try:
    from utils.datasets import LoadImages, LoadImagesAndLabels
    from utils.general import plot_one_box, xywh2xyxy
except:
    sys.path.append('..')
    from utils.datasets import LoadImages, LoadImagesAndLabels
    from utils.general import plot_one_box, xywh2xyxy


class CheckDataset(object):
    def __init__(self, dp):
        self.dp = dp
        # self.colors = [[random.randint(0, 255) for _ in range(3)] for __ in range(100)]
        # self.colors = ['#009966', '#EE3B3B']
        self.colors = [[102, 153, 0], [59, 59, 238]]
        self.names = ['crosswalk', 'guide_arrows']

    def __call__(self, trte='test', cf=False):
        p = f'{self.dp}/images/{trte}'

        imgs = [f'{self.dp}/images/{trte}/{x}' for x in os.listdir(p) if x.endswith('.jpg')]
        imgs = sorted(imgs)
        for i, imgp in enumerate(imgs):
            stem = Path(imgp).stem
            labelp = f'{self.dp}/labels/{trte}/{Path(imgp).stem}.txt'
            img = cv2.imread(imgp)
            h, w, c = img.shape

            with open(labelp, 'r') as f:
                label = f.readlines()
            label = np.array([x.split() for x in label], dtype=np.float32)

            classes = label[:, 0]
            bboxes = label[:, 1::]
            bboxes = xywh2xyxy(bboxes)
            for j in range(len(label)):
                cls = classes[j]
                bbox = bboxes[j]
                bbox[0] *= w
                bbox[1] *= h
                bbox[2] *= w
                bbox[3] *= h
                plot_one_box(bbox, img, label=f'{self.names[int(cls)]}', color=self.colors[int(cls)])

            print(f'imgs: {len(imgs)} stem: {stem} img_shape: {img.shape} lb: {label}')
            # cr = np.any(label[:, 0] == 1)
            crit = 'fogged' in stem if cf else True
            if crit:
                cv2.imshow('xx', img)
                if cv2.waitKey(0) == ord('q'):
                    exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset-path', type=str,
                        help='path to dataset need to be checked. such as: xxx/train_data_v5_format',
                        default='/home/zzd/datasets/crosswalk/train_data_v5_format')
    parser.add_argument('-trte', '--train_or_test', type=str,
                        help='specify check train or test of YOLOv5 format',
                        default='train')
    parser.add_argument('-cf', '--only-check-fog', action='store_true', help='only check fogged dataset')

    opt = parser.parse_args()
    cd = CheckDataset(dp=opt.dataset_path)
    cd(trte=opt.train_or_test, cf=opt.only_check_fog)
