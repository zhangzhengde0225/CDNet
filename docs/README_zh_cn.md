
#### [English](https://github.com/zhangzhengde0225/CDNet) | 简体中文

如果本项目对你有帮助，请点击项目右上角star支持一下和引用论文。

# CDNet

这个项目是论文的《CDNet: 一个基于YOLOv5的在Jetson Nano上实时、鲁棒的斑马线检测网络》的代码、数据集和教程。

CDNet (Crosswalk Detection Network) 是车载摄像头视野下检测斑马线（人行横道）和分析车辆过线行为的具体实现。

![GA](https://zhangzhengde0225.github.io/images/CDNet_GA.jpg)

Fig.1 图形摘要

# Highlights
+ 实现了斑马线检测和车辆过线行为检测。
+ 在特定任务中准确率和速度超过原生YOLOv5。
+ 在阴天、晴天、雨天、夜间等真实复杂场景中实现高鲁棒性。
+ 在 Jetson nano 边缘计算设备上实现实时检测 (33.1 FPS)。
+ 提供了标注好的斑马线数据集，共计6868张图 。

# 贡献Contribution

+ 注意力机制网络改进网络，提升精度，略微降低速度：SENet (Squeeze-and-Excitation Network)
+ 负样本训练，提升精度，速度不变: NST (Negative Samples Training)
+ 感兴趣区域，提升速度，精度下降：ROI (Region Of Interest)
+ 滑动感受野短时向量记忆算法，迁移斑马线检测任务到汽车过线行为检测任务，提升精度，速度不变：SSVM (Slide receptive field Short-term Vectors Memory)
+ 合成雾增强算法，增强数据集，适应雾天，提升精度，速度不变：SFA (Synthetic Fog Augment)

# 安装Installation
安装CDNet代码并配置环境，请查看：
[docs/INSTALL.md](https://github.com/zhangzhengde0225/CDNet/blob/master/docs/INSTALL.md)

# 模型库Model Zoo
模型库请查看：
[docs/MODEL_ZOO.md](https://github.com/zhangzhengde0225/CDNet/blob/master/docs/MODEL_ZOO.md)

# 数据集Datasets
下载训练集和测试集，请查看：
[docs/DATASETS.md](https://github.com/zhangzhengde0225/CDNet/blob/master/docs/DATASETS.md)

# 快速开始Quick Start
## 训练Train

安装CDNet代码，配置环境和下载数据集后，输入代码训练：
```
python train.py --trainset_path </path/to/trainset/folder>
(such as: /home/xxx/datasets/train_data_yolov5_format) 
```
训练结果和权重将保存在 runs/xxx/ 目录中。

主要的可选参数：
```
--device "0, 1"  # cpu or gpu id, "0, 1" means use two gpu to train.
--img-size 640 
--batch-size 32 
--epochs 100 
--not-use-SE  # use original YOLOv5 which not SE-module embedded if there is the flag
```

## 推理Inference
逐图或视频检测斑马线并分析车辆过线行为，输入代码：
```
python detect.py
```

主要的可选参数：
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

关于安装和数据集的更多详情请参考：
[docs/INSTALL.md](https://github.com/zhangzhengde0225/CDNet/blob/master/docs/INSTALL.md) and [docs/DATASETS.md](https://github.com/zhangzhengde0225/CDNet/blob/master/docs/DATASETS.md).

## 合成雾增强Fogging Augment
如果你想通过合成雾算法增加数据集，只需运行：
```
python fog_augment.py
```
更多细节请查看fog_augment.py和/scripts/synthetic_fog.py中的源代码

# 结果Results

![Results](https://zhangzhengde0225.github.io/images/CDNet_Results.jpg)
Fig.2 与原生YOLOv5性能对比Performance compared to YOLOv5.

**与原生YOLOv5相比，检测尺寸为640时，CDNet 在 Jetson nano 上提高了5.13%的F1分数和10.7FPS的速度，**
**检测尺寸为288时，提升为13.38%的F1分数和13.1FPS。**

# 贡献者Contributors
CDNet的作者是: Zhengde Zhang, Menglu Tan, Zhicai Lan, Haichun Liu, Ling Pei and Wenxian Yu。

目前，CDNet由
Zhengde Zhang (drivener@163.com)负责维护。

如果您有任何问题，请随时与我们联系。

Zhengde Zhang的学术主页: [zhangzhengde0225.github.io](https://zhangzhengde0225.github.io).

# 致谢Acknowledgement

我们非常感谢
[yolov5](https://github.com/ultralytics/yolov5) 
项目提供的目标检测算法基准。

我们非常感谢
[tensorrtx](https://github.com/wang-xinyu/tensorrtx)
项目提供的模型模署到Jetson nano上的技术。

# 链接Links

B站视频样例：[https://www.bilibili.com/video/BV1qf4y1B7BA](https://www.bilibili.com/video/BV1qf4y1B7BA)

阅读论文全文：[https://rdcu.be/cHuc8](https://rdcu.be/cHuc8)

下载论文全文：[https://doi.org/10.1007/s00521-022-07007-9](https://doi.org/10.1007/s00521-022-07007-9)

CSDN项目简介：[http://t.csdn.cn/Cf7c7](http://t.csdn.cn/Cf7c7)

如果对您有帮助，请为点击项目右上角的star支持一下或引用论文。

# 引用Citation
```
@article{CDNet,
author={Zheng-De Zhang, Meng-Lu Tan, Zhi-Cai Lan, Hai-Chun Liu, Ling Pei and Wen-Xian Yu},
title={CDNet: a real-time and robust crosswalk detection network on Jetson nano based on YOLOv5},
Journal={Neural Computing and Applications}, 
Year={2022},
DOI={10.1007/s00521-022-07007-9},
}
```


# 许可License
CDNet可免费用于非商业用途，并可在这些条件下重新分发。 如需商业咨询，请发送电子邮件至
drivener@163.com，我们会将详细协议发送给您。









