# CDNet

This repository is the codes, datasets and tutorials for the paper 
"CDNet: a real-time and robust crosswalk detection network on Jetson nano based on YOLOv5".



CDNet (Crosswalk Detection Network) is a specific implementation of crosswalk (zebra crossing) detection and vehicle crossing behavior analysis under the vision of vehicle-mounted camera. 

![graphical abstract](https://github.com/zhangzhengde0225/CDNet/blob/master/data/graphical_abstract.jpg)

Fig.1 Graphical abstract.

# Highlights
+ A crosswalk detection and vehicle crossing behavior detection network is realized.
+ The accuracy and speed exceed YOLOv5 in the specific task.
+ High robustness in real complex scenarios such as in cloudy, sunny, rainy and at night is achieved.
+ Real-time detection (33.1 FPS) is implemented on Jetson nano edge-computing device.
+The datasets, tutorials and source codes are available on GitHub.
  

On the basis of [YOLOv5](https://github.com/ultralytics/yolov5), the following technologies are proposed to improve the speed and accuracy :

+ SENet (Squeeze-and-Excitation Network)
+ NST (Negative Samples Training)
+ ROI (Region Of Interest)
+ SSVM (Slide receptive field Short-term Vectors Memory)
+ SFA (Synthetic Fog Augment)

# News!
+ Sep 2021: [1.1.0 version](https://github.com/zhangzhengde0225/CDNet) is released! The synthetic fog algorithm is implemented, 
  the datasets has been enhanced to 6160 images, which half of the images is fogged. The fogging source code is released.
+ Jan 2021: [1.0.1 version](https://github.com/zhangzhengde0225/CDNet) of CDNet is released! 
  With a detection speed of 33.1 FPS on Jetson nano, it obtained an average F1 score of 94.72% in crossing bebahior detection under the complex scenarios.

# Installation
Get CDNet code and configure the environment, please check out [docs/INSTALL.md](https://github.com/zhangzhengde0225/CDNet/blob/master/docs/INSTALL.md)

# Model Zoo
Please check out [docs/MODEL_ZOO.md](https://github.com/zhangzhengde0225/CDNet/blob/master/docs/MODEL_ZOO.md)

# Datasets
Download trainsets and testsets, please check out [docs/DATASETS.md](https://github.com/zhangzhengde0225/CDNet/blob/master/docs/DATASETS.md)

# Quick Start
## Training

Once you get the CDNet code, configure the environment and download the dataset, juse type:
```
python train.py --trainset_path </path/to/trainset/folder>
(such as: /home/xxx/datasets/train_data_yolov5_format) 
```
The training results and weights will be saved in runs/expxx/ directory.

The main optional arguments:
```
--device "0, 1"  # cpu or gpu id, "0, 1" means use two gpu to train.
--img-size 640 
--batch-size 32 
--epochs 100 
--not-use-SE  # use original YOLOv5 which not SE-module embedded if there is the flag
```

## Inference
Detect the crosswalk image by image and analyze the vehicle crossing behavior. 
```
python detect.py
```

The main optional arguments:
```
--source example/images  # images dir
--output example/output  # output dir
--img-size 640  # inference model size
--device "0"   # use cpu or gpu(gpu id)
--plot-classes ["crosswalk"]  # plot classes
--field-size 5  # the Slide receptive field size of SSVM 
--not-use-ROI  # not use roi for accelerate inference speed if there is the flag
--not-use-SSVM  # not use ssvm method for analyse vehicle crossing behavior if there is the flag
```

For more details, please refer to [docs/INSTALL.md](https://github.com/zhangzhengde0225/CDNet/blob/master/docs/INSTALL.md) and [docs/DATASETS.md](https://github.com/zhangzhengde0225/CDNet/blob/master/docs/DATASETS.md).

## Fogging Augment
If you want to augment datasets by synthetic fog algorithm, just run:
```
python fog_augment.py
```
For more details, please view the source code in fog_augment.py and /scripts/synthetic_fog.py

# Results

![Detection results compare to YOLOv5](https://github.com/zhangzhengde0225/CDNet/blob/master/data/Detection%20results%20compare%20to%20YOLOv5.jpg)

Fig.2 Performance compared to YOLOv5.

**CDNet improves the score for 5.13 points and speed for 10.7 FPS on Jetson nano for detection size of 640 compared to YOLOv5.**

**For detection size of 288, the improvements are 13.38 points and 13.1 FPS.**


# Contributors
CDNet is authored by Zhengde Zhang, Menglu Tan, Zhicai Lan, Haichun Liu, Ling Pei and Wenxian Yu.

Currently, it is maintained by Zhengde Zhang (drivener@163.com).

Please feel free to contact us if you have any question.

Homepage of Zhengde Zhang: [zhangzhengde0225.github.io](https://zhangzhengde0225.github.io).

# Citation
```
@article{Zheng-De Zhang, Meng-Lu Tan, Zhi-Cai Lan, Hai-Chun Liu, Ling Pei and Wen-Xian Yu.
CDNet: a real-time and robust crosswalk detection network on Jetson nano based on YOLOv5.
Neural Computing and Applications, 2022.
DOI: 10.1007/s00521-022-07007-9
}
```
Please star this project if it helps you.

Paper link: [https://doi.org/10.1007/s00521-022-07007-9](https://doi.org/10.1007/s00521-022-07007-9)


# License
CDNet is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, please drop an e-mail at drivener@163.com. We will send the detail agreement to you.





