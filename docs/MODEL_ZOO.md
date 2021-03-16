# MODEL ZOO

## Pytorch model

|Model|Img Size|F1 score|CPU time (ms)|Nano time (ms)|Pytorch weights|TensorRT engine|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|YOLOv5m (base) |288x192|81.34|189|43|[Download](link)|[Download](link)
|YOLOv5m+NST    |288x192|89.03|189|43|
|YOLOv5m+NST+SE |288x192|86.41|205|  |[Download](link)
|YOLOv5m+NST+ROI|288x192|81.27|74 |26|
|
|YOLOv5m+NST+ROI+SSVM|288x192|93.70|74|26|
|YOLOv5m+NST+SE+ROI+SSVM (CDNet)|288X192|94.72|86|[model](link)

**Notes**

CPU inference time measures per image with **i7-4770HQ**@2.2GHz CPU on Macbook Pro 2014.

Nano inference time measures per image (FP16) on **Jetson Nano** (4GB) with JetPack-4.4, cuda-10.2 and TensorRT-7.1.3.

We also tested the GPU inference time on **RTX 3080** using [TensorRT](https://developer.nvidia.com/tensorRt), and the result shows the it is about 3.1ms per image (FP16).

Note that TensorRT is optimized for specific TensorRT version, CUDA version and hardware, so the engine downloaded from this page can only be used on Jetson Nano with specific package installed.

If you want to export the TensorRT engine from the pytorch weights for specific hardware and package, please refers to [here](https://github.com/wang-xinyu/tensorrtx).