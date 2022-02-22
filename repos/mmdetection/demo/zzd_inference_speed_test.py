# Copyright (c) OpenMMLab. All rights reserved.
import os
import asyncio
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()

    path = '../../example/images'
    imgs = sorted([f'{path}/{x}' for x in os.listdir(path) if x.endswith('jpg')])
    args.img = imgs

    ck_dict = dict(mask_rcnn='zzd_mask_rcnn_r50_fpn_1x_crosswalk/epoch_12.pth',
                   faster_rcnn='zzd_faster_rcnn_r50_fpn_1x_crosswalk/epoch_12.pth',
                   yolox='zzd_yolox_m_8x8_300e_crosswalk/epoch_100.pth',
                   swin='zzd_mask_rcnn_swin-t-p4-w7_fpn_1x_crosswalk/epoch_12.pth')

    model = 'mask_rcnn'
    model = 'faster_rcnn'
    model = 'yolox'
    model = 'swin'
    args.config = f'./configs/{model}/{ck_dict[model].split("/")[0]}.py'
    args.checkpoint = f'work_dirs/{ck_dict[model]}'
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    t = []
    for i, img in enumerate(args.img):
        result, st = inference_detector(model, img)  # spent time
        print(f'{i:>2} spent time: {st}')
        if i != 0:
            t.append(st)
    t = np.array(t)
    print(f'total time : {np.sum(t):.4f}s mean: {np.mean(t)*1000:.2f}ms FPS: {1/np.mean(t):.2f}')
    # show the results
    # show_result_pyplot(model, args.img, result, score_thr=args.score_thr)


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(model, args.img, result[0], score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
