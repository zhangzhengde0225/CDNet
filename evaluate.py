"""
evaluate model performance on test set
(1) 指定detect后的目标文件夹，获取评估结果
(2) 指定不存在的检测文件夹，需要指定权重、ROI等一系列参数，获取评估结果
"""
import os
import argparse
from pathlib import Path
import re

import detect
from scripts import post
from scripts.get_confusing_matrix import GetConfusingMatrix


class CDNetEvaluator(object):
	def __init__(self):
		self.gcm = GetConfusingMatrix()

	def __call__(self, opt):
		exps = sorted(os.listdir('./runs'))
		assert len(exps), f'Experiments not exists, please train model first'
		q = "\n".join([f'[{i:>2}] {x}' for i, x in enumerate(exps)])
		ipt = input(
			f'Existing experiments:\n{q}\nPlease select which model to evaluate\n'
			f'(0 for model0, 0,1,2 for model0,1,2): ')
		ipt = '0' if ipt == '' else ipt
		indexes = [int(x) for x in ipt.split(',')] if ',' in ipt else [int(ipt)]
		for idx in indexes:
			print('\n')
			exp = exps[idx]
			opt.weights = f'./runs/{exp}/weights/best.pt'

			roi = int(opt.use_roi)
			ssvm = int(opt.use_ssvm)
			tp = f'{Path(opt.source).parent}/{exp}_sz{opt.img_size}_ROI{roi}_SSVM{ssvm}{opt.fog}'  # target path

			flag = 0  # 0: detect and evaluate, 1: direct evaluate
			if os.path.exists(tp):
				ipt = input(f'Output_path: {tp} exists, \nevaluate without detect [YES/no]: ')
				if ipt in ['Yes', 'y', 'Y', 'yes', '']:
					flag = 1
				elif ipt in ['No', 'N', 'n']:
					flag = 0
				else:
					raise NotImplementedError(f'input error: {ipt}')
			if flag == 0:
				opt.output = tp
				print(f'opt: {opt}')
				dp = post.DmPost(opt)
				detect.run(opt, dp)

			ret = self.gcm.analyse(thresh=0.5, v5_out=tp, pn_dir=Path(opt.source).parent)
			time, ret = self.read_time_and_fps(tp=tp)

	def read_time_and_fps(self, tp):
		with open(f'{tp}/detect.log', 'r') as f:
			log = f.readlines()
		time, fps = re.findall(f'\d+.\d+', log[-1])
		time = time[:-1]
		fps = fps[:-3]
		return time, fps

def read_and_update_opt(eval_opt):
	det_opt = detect.get_opt()
	det_opt.source = eval_opt.source
	det_opt.output = eval_opt.output
	det_opt.not_use_ROI = eval_opt.not_use_ROI
	det_opt.not_use_SSVM = eval_opt.not_use_SSVM
	det_opt.img_size = eval_opt.img_size

	det_opt.use_roi = not det_opt.not_use_ROI
	det_opt.use_ssvm = not det_opt.not_use_SSVM
	return det_opt


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--source', type=str, default='example/images', help='source')
	parser.add_argument('--output', type=str, default='example/output', help='output folder')  # output folder
	parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
	parser.add_argument('--not-use-ROI', action='store_true',
						help='not use roi for accelerate inference speed if there is the flag')
	parser.add_argument('--not-use-SSVM', action='store_true',
						help='not use ssvm method for analyse vehicle crossing behavior if there is the flag')
	opt = parser.parse_args()

	# opt.source = "/home/zzd/datasets/crosswalk/testsets_1770/Images"
	opt.source = "/home/zzd/datasets/crosswalk/testsets_1770/fogged_Images"
	opt = read_and_update_opt(opt)
	opt.fog = '_fogged' if 'fogged' in opt.source else ''

	CDNetE = CDNetEvaluator()
	CDNetE(opt)

