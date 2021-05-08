# MODEL ZOO

## Pytorch model

|Model|Detect Size|Speed (FPS)|F1 Score (%)|Pytorch Model|TensorRT Engine|
|:---:|:---:|:---:|:---:|:---:|:---:|
|YOLOv5m (baseline)|288x192|20.0|81.34|[Download](https://pan.baidu.com/s/1FCW5urZynjXjteR1E_2XuA) passwd: **duap**|[Download](link)
|SE-YOLOv5m+NST+ROI+SSVM (CDNet)|288X192|33.1|94.72|[Download](https://pan.baidu.com/s/18p2TS-x830X7IUAVsdihow) passwd: **1cpp**|[Download](link)

**Notes**

Detection speed measured on **Jetson Nano** (4GB) with JetPack-4.4, cuda-10.2 and TensorRT-7.1.3.

We also tested the GPU inference time on **RTX 3080** using [TensorRT](https://developer.nvidia.com/tensorRt), and the result shows that the speed is about 323 FPS (3.1 ms) per image.

Note that TensorRT is optimized for specific TensorRT version, CUDA version and hardware, so the engine downloaded from this page can only be used on Jetson Nano with specific package installed.

If you want to export the TensorRT engine from the pytorch weights for specific hardware and package, please refers to [here](https://github.com/wang-xinyu/tensorrtx).