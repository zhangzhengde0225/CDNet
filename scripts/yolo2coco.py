"""
转换训练集，YOLOv5 format to COCO format
"""
import os, sys
sys.path.append(f'/home/zzd/PycharmProject/damei')
import damei as dm

sp = '/home/zzd/datasets/crosswalk/fogged_train_data_v5_format'
tp = "/home/zzd/datasets/crosswalk/fogged_train_data_coco_format"
dm.tools.yolo2coco(sp=sp, tp=tp)

