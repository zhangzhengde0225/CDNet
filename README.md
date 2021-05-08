Paper CDNet: [link](link)

This repository represents Crosswalk Detection Network (CDNet), which is a specific implementation of crosswalk (zebra crossing) detection and vehicle crossing behavior analysis under the vision of vehicle-mounted camera. 

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

# News!
+ Jan 2021: [1.0.1 version](https://github.com/zhangzhengde0225/CDNet) of CDNet is released! 
  With a detection speed of 33.1 FPS on Jetson nano, it obtained an average F1 score of 94.72% in the complex scenarios.

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
python train.py --trainset_path /path/to/trainset/folder
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


# Results

![Detection results compare to YOLOv5](https://github.com/zhangzhengde0225/CDNet/blob/master/data/Detection%20results%20compare%20to%20YOLOv5.jpg)

**The square dots denote detection size of 640, while triangle dots denote 288.**
**The detection speed measured on the NVIDIA edge computing device Jetson nano.**

# Contributors
CDNet is authored by Zhengde Zhang, Menglu Tan, Zhicai Lan, Haichun Liu, Ling Pei and Wenxian Yu.

Currently, it is maintained by Zheng-De Zhang (drivener@163.com).

# Citation
```
xxx
```

# License
CDNet is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, please drop an e-mail at drivener@163.com. We will send the detail agreement to you.





