"""
检查coco数据集
"""
import os, sys
import argparse

sys.path.append(f'/home/zzd/PycharmProject/damei')
import damei as dm

# parser = argparse.ArgumentParser()
# parser.add_argument('--json_path', )


dp = "/home/zzd/datasets/crosswalk/fogged_train_data_coco_format"
jp = f'{dp}/annotations/instances_train2017.json'
# jp = "/home/zzd/datasets/mscoco/annotations/instances_train2017.json"
dm.tools.check_coco(json_path=jp)









