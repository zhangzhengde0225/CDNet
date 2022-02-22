import os
import subprocess

cfg = "./configs/mask_rcnn/zzd_mask_rcnn_r50_fpn_1x_crosswalk.py"
cfg = "./configs/faster_rcnn/zzd_faster_rcnn_r50_fpn_1x_crosswalk.py"
cfg = "./configs/centernet/zzd_centernet_resnet18_dcnv2_140e_crosswalk.py"
cfg = './configs/swin/zzd_mask_rcnn_swin-t-p4-w7_fpn_1x_crosswalk.py'
cfg = './configs/yolox/zzd_yolox_m_8x8_300e_crosswalk.py'

code = f'python tools/train.py {cfg}'
# subprocess.call(code)
os.system(code)
