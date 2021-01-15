Paper CDNet: [link](link)

This repository represents Crosswalk Detection Network (CDNet), which is a specific implementation of crosswalk (zebra crossing) detection and vehicle crossing behavior analysis under the vision of vehicle-mounted camera. 

The project solves the problem of high robustness detection and analysis of crosswalkd in complex scenes, such as reflection after rain, occluded, distorted, truncated, damaged, view blocked, dazzling, partially lost and so on.

[figure]


On the basis of [YOLOv5](https://github.com/ultralytics/yolov5), the following technologies are proposed and applied to further improve the speed and accuracy of low computing power devices in crosswalk detection task:

+ SENet (Squeeze-and-Excitation Network)
+ NST (Negative Samples Training)
+ ROI (Region Of Interest)
+ SSVM (Slide receptive field Short-term Vectors Memory)

# News!
+ Jan 2021: [1.0.1 version]() of CDNet is released! It achieves 94.72% F1_score on [Crosswalk validation dataset]() with 86 ms (on i7-4770HQ CPU) and 3.1 ms (on RTX 3080 GPU) inference time

# Results


# Installation
Please check out [docs/INSTALL.md](https://github.com/zhangzhengde0225/CDNet/blob/master/docs/INSTALL.md)

# Model Zoo
Please check out [docs/MODEL_ZOO.md](https://github.com/zhangzhengde0225/CDNet/blob/master/docs/MODEL_ZOO.md)

# Datasets
Download trainsets and testsets, please check out [docs/DATASETS.md](https://github.com/zhangzhengde0225/CDNet/blob/master/docs/DATASETS.md)


# Enviroments
python 3.6 or later with all requirement.txt denpendencies installed, including pytorch>=1.6. To install run:
```python
pip install -r requirements.txt
```
# Quick Start
**Training**: Trian from scratch
```python
python train.py --
```
**Inference**: Inference images
```python
python detect.py --
```
**Validation**: Repreduce the results on paper
```python
python validation.py --
```

For more details, please refer to [GETTING_STARTED.md]().


# Contributors
CDNet is authored by Zhengde Zhang, Menglu Tan, Zhicai Lan, Haichun Liu, Ling Pei and Wenxian Yu, Zhengde Zhang and Wenxin Yu is the corresponding author.

Currently, it is maintained by Zhengde Zhang (zhangzhengde@mail.sjtu.edu.cn)

# Citation


# License





