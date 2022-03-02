
今天带来一个基于改进YOLOv5的斑马线和汽车过线行为检测算法，文章提出多项tricks，数据集和复现代码开源！

标题: CDNet: A Real-Time and Robust Crosswalk Detection Network on Jetson Nano Based on YOLOv5
CDNet: 一个基于YOLOv5的在Jetson Nano上实时、鲁棒的斑马线检测网络
作者：Zheng-De Zhang, Meng-Lu Tan, Zhi-Cai Lan, Hai-Chun Liu, Ling Pei, Wen-Xian Yu
时间：Feb, 2022
期刊：Neural Computing & Applications, IF 5.6

图形摘要：
 

摘要：
在复杂场景和有限计算能力下实现实时、鲁棒的斑马线（人行横道）检测是当前智能交通管理系统（ITMS）的重要难点之一。有限的边缘计算能力和多云、晴天、雨天、雾天和夜间等真实复杂的场景同时对这项任务提出了挑战。本研究提出基于改进YOLOv5的人行横道检测网络（CDNet），实现车载摄像头视觉下快速准确的人行横道检测，并在Jetson nano设备上实现实时检测。强大的卷积神经网络特征提取器用于处理复杂环境，网络中嵌入了squeeze-and-excitation（SE）注意力机制模块，使用负样本训练（NST）方法提高准确率，利用感兴趣区域（ROI）算法进一步提高检测速度，提出了一种新的滑动感受野短时向量记忆（SSVM）算法来提高车辆交叉行为检测精度，使用合成雾增强算法允许模型适应有雾的场景。最后，在 Jetson nano 上以 33.1 FPS 的检测速度，我们在上述复杂场景中获得了 94.83% 的平均 F1 分数。对于晴天和阴天等更好的天气条件，F1 分数超过 98%。该工作为人工神经网络算法优化方法在边缘计算设备上的具体应用提供了参考，发布的模型为智能交通管理系统提供了算法支持。

贡献：
+ 注意力机制网络改进网络，提升精度，略微降低速度：SENet (Squeeze-and-Excitation Network)
+ 负样本训练，提升精度，速度不变: NST (Negative Samples Training)
+ 感兴趣区域，提升速度，精度下降：ROI (Region Of Interest)
+ 滑动感受野短时向量记忆算法，迁移斑马线检测任务到汽车过线行为检测任务，提升精度，速度不变：SSVM (Slide receptive field Short-term Vectors Memory)
+ 合成雾增强算法，增强数据集，适应雾天，提升精度，速度不变：SFA (Synthetic Fog Augment)
+ 斑马数据集：标注好的，车载摄像头视角下的，共计6868张图像。
+ 复现代码：见github项目主页


数据集、复现代码：https://github.com/zhangzhengde0225/CDNet
视频样例：https://www.bilibili.com/video/BV1qf4y1B7BA
阅读全文：https://rdcu.be/cHuc8
下载全文：https://doi.org/10.1007/s00521-022-07007-9
如果该项目对你有帮助，请点击github项目右上角star收藏和引用本论文，谢谢~