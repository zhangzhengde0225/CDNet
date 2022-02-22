import os
import subprocess

img = 'demo/demo.jpg'
# cfg = './configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# checkpoint = './checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

img = '../../example/images/filename_00038.jpg'
cfg = './configs/mask_rcnn/zzd_mask_rcnn_r50_fpn_1x_crosswalk.py'
checkpoint = 'work_dirs/zzd_mask_rcnn_r50_fpn_1x_crosswalk/epoch_12.pth'

code = f'python demo/image_demo.py {img}' \
	f' {cfg} {checkpoint} --device cpu'
print(code)
# subprocess.call(code)
os.system(code)
