[![Datasets](https://img.shields.io/static/v1?label=Download&message=datasets&color=green)](
https://github.com/zhangzhengde0225/CDNet/blob/master/docs/DATASETS.md)
[![Datasets](https://img.shields.io/static/v1?label=Download&message=source_code&color=orange)](
https://github.com/zhangzhengde0225/CDNet/archive/refs/heads/master.zip)
[![Open issue](https://img.shields.io/github/issues/zhangzhengde0225/CDNet)](https://github.com/zhangzhengde0225/CDNet/issues)
[![Stars](https://img.shields.io/github/stars/zhangzhengde0225/CDNet)](https://github.com/zhangzhengde0225/CDNet)
#### English | [简体中文](https://github.com/zhangzhengde0225/CDNet/blob/master/docs/README_zh_cn.md)

Please **star this project** in the upper right corner and **cite this paper** blow 
if this project helps you. 

# CDNet

This repository is the codes, datasets and tutorials for the paper 
"CDNet: a real-time and robust crosswalk detection network on Jetson nano based on YOLOv5".

CDNet (Crosswalk Detection Network) is a specific implementation of crosswalk (zebra crossing) detection and vehicle crossing behavior analysis under the vision of vehicle-mounted camera. 

![GA](https://zhangzhengde0225.github.io/images/CDNet_GA.jpg)

Fig.1 Graphical abstract.

# Highlights
+ A crosswalk detection and vehicle crossing behavior detection network is realized.
+ The accuracy and speed exceed YOLOv5 in the specific task.
+ High robustness in real complex scenarios such as in cloudy, sunny, rainy and at night is achieved.
+ Real-time detection (33.1 FPS) is implemented on Jetson nano edge-computing device.
+The datasets, tutorials and source codes are available on GitHub.
  
# Contribution

+ SENet (Squeeze-and-Excitation Network), F1 score up, speed slightly down
+ NST (Negative Samples Training), F1 score up, speed invariant
+ ROI (Region Of Interest), F1 score down, speed up
+ SSVM (Slide receptive field Short-term Vectors Memory), transfer crosswalk detection task into vehicle crossing behavior task, 
  F1 score up, speed invariant
+ SFA (Synthetic Fog Augment), dataset augment, adapt to foggy weather, F1 score up, speed invariant

# Installation
Get CDNet code and configure the environment, please check out [docs/INSTALL.md](https://github.com/zhangzhengde0225/CDNet/blob/master/docs/INSTALL.md)

# Model Zoo
Please check out [docs/MODEL_ZOO.md](https://github.com/zhangzhengde0225/CDNet/blob/master/docs/MODEL_ZOO.md)

# Datasets
Download trainsets and testsets, please check out [docs/DATASETS.md](https://github.com/zhangzhengde0225/CDNet/blob/master/docs/DATASETS.md)

# Quick Start
## Train

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

![Results](https://zhangzhengde0225.github.io/images/CDNet_Results.jpg)

Fig.2 Performance compared to YOLOv5.

**CDNet improves the score for 5.13 points and speed for 10.7 FPS on Jetson nano for detection size of 640 compared to YOLOv5.**

**For detection size of 288, the improvements are 13.38 points and 13.1 FPS.**


# Contributors
CDNet is authored by Zhengde Zhang, Menglu Tan, Zhicai Lan, Haichun Liu, Ling Pei and Wenxian Yu.

Currently, it is maintained by Zhengde Zhang (drivener@163.com).

Please feel free to contact us if you have any question.

The Academic homepage of Zhengde Zhang: [zhangzhengde0225.github.io](https://zhangzhengde0225.github.io).

# Acknowledgement

This work was supported by the National Natural
Science Foundation of China [Grant Numbers: 61873163]. 

We acknowledge the Center for High Performance Computing at
Shanghai Jiao Tong University for providing computing resources.

We are very grateful to the 
[yolov5](https://github.com/ultralytics/yolov5) project
for the benchmark detection algorithm.

We are very grateful to the 
[tensorrtx](https://github.com/wang-xinyu/tensorrtx) project
for the deployment techniques to the Jetson nano.

# Links
Detect Video Samples：[https://www.bilibili.com/video/BV1qf4y1B7BA](https://www.bilibili.com/video/BV1qf4y1B7BA)

Read Full Text of This Paper：[https://rdcu.be/cHuc8](https://rdcu.be/cHuc8)

Download Full Text of this Paper：[https://doi.org/10.1007/s00521-022-07007-9](https://doi.org/10.1007/s00521-022-07007-9)

Project Introduction on CSDN：[http://t.csdn.cn/Cf7c7](http://t.csdn.cn/Cf7c7)

If it is helps you, 
please star this project in the upper right corner and cite this paper blow.

# Citation
```
@article{CDNet,
author={Zheng-De Zhang, Meng-Lu Tan, Zhi-Cai Lan, Hai-Chun Liu, Ling Pei and Wen-Xian Yu},
title={CDNet: a real-time and robust crosswalk detection network on Jetson nano based on YOLOv5},
Journal={Neural Computing and Applications}, 
Year={2022},
DOI={10.1007/s00521-022-07007-9},
}
```

# License
CDNet is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, please drop an e-mail at drivener@163.com. We will send the detail agreement to you.





