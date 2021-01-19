"""
针对crosswalk 1770测试集，计算混淆矩阵
"""

import os
import numpy as np

import time
from PIL import Image
from pathlib import Path
import re


class GetConfusingMatrix(object):
	"""
	for crosswalk detection.
	"""
	def __init__(self, thresh=0.9, img_suffix='jpg'):
		self.thresh = thresh
		self.img_suffix = img_suffix
		self.classes = ['crosswalk']

	def analyse(self, weight_path=None, need_infer=False, thresh=None, v5_out=None, pn_dir=None):
		print(f"{Path(v5_out).stem:-^80}")

		self.thresh = thresh if thresh is not None else self.thresh

		gt_data = self.get_ground_truth(pn_dir)  # array [n, 2] 2:filename, cls
		# print(gt_data.shape)

		# 获取detect data
		dt_data = self.get_detect_data(need_infer=need_infer, weight_path=weight_path, v5_out=v5_out)
		# array [n_files, 100, 3] 100: 同一张图最大检测到的目标数目，3: filename, cls, conf
		assert len(self.classes) == 1
		for i, cls in enumerate(self.classes):
			gt = gt_data[gt_data[:, 1] == str(i)]
			dt = dt_data[np.where(dt_data[:, :, 1] == str(i))].astype(np.str)  # (n, 7)  # 筛选出该类
			new_dt = np.zeros((dt_data.shape[0], dt_data.shape[2])).astype(np.str)  # (1770, 7)
			for j, file in enumerate(dt_data[:, 0, 0]):  # file 是文件名
				# print(j, file, dt[:, 0].shape, dt_data.shape, new_dt.shape)
				if file in dt[:, 0]:
					new_dt[j] = dt[dt[:, 0] == file][0]  # 取出第0个
				else:
					new_dt[j] = np.array([file, '-2', '-2', '-2', '-2', '-2', '-2'])
			ret = self.cal_TPFPTNFN(cal_cls=self.classes[i], dt_data=new_dt, gt_data=gt)
		return ret

	def cal_TPFPTNFN(self, cal_cls, dt_data, gt_data):
		"""
		计算混淆矩阵。
		:param cal_cls: 需要计算的类别名称
		:param dt_data: 检测结果：[n, 7] n: 所有检测的图，7: img, cls, conf, xc, yc, w, h
		:param gt_data: 真值
		:return:
		"""
		TP_count, FP_count, TN_count, FN_count = 0, 0, 0, 0
		gt_filenames = gt_data[:, 0].tolist()
		for i in range(dt_data.shape[0]):
			filename, cls, conf = dt_data[i, :3]
			if filename in gt_filenames:
				gt = True
			else:
				gt = False
			if float(conf) >= self.thresh:
				dt = True
			else:
				dt = False

			if gt and dt:
				TP_count += 1
			elif gt and not dt:
				FN_count += 1
			elif not gt and dt:
				FP_count += 1
			elif not gt and not dt:
				TN_count += 1
			else:
				raise Exception(f'ERROR, gt:{gt} dt:{dt}')
		# print(filename, gt, dt)
		# if i > 100:
		# break
		confusing_matrix = np.array([TP_count, FP_count, FN_count, TN_count]).astype(np.int)
		# print(confusing_matrix)
		# exit()

		# confusing_matrix = confusing_matrix/len(data)
		# print(confusing_matrix)
		alpha = 1e-10

		ACC = (TP_count + TN_count) / (TP_count + TN_count + FP_count + FN_count+alpha)  # 准确率
		PPV = (TP_count) / (TP_count + FP_count+alpha)  # 精确率
		TPR = (TP_count) / (TP_count + FN_count+alpha)  # 召回率
		TNR = (TN_count) / (TN_count + FP_count+alpha)  # 特异度
		F1_score = 2 * PPV * TPR / (PPV + TPR+alpha)

		cm = confusing_matrix
		string1 = f"{'cls':^15}{'ALL':<5}{'TP':<5}{'FP':<5}{'FN':<5}{'TN':<5}\n" \
				  f"{cal_cls:^15}{len(dt_data):<5}{cm[0]:<5}{cm[1]:<5}{cm[2]:<5}{cm[3]:<5}"
		# print(f"ALL   TP    FP   FN  TN")
		# print(f"{len(dt_data)} {confusing_matrix}")
		print(string1)
		string2 = f'Thresh: {self.thresh} Accuracy: {ACC*100:.2f}% Precision: {PPV*100:.2f}% Recall: {TPR*100:.2f}% ' \
				  f'Specificity: {TNR*100:.2f}% F1_score: {F1_score*100:.2f}%'
		print(string2)
		pnts3 = f"{len(dt_data):<5}{cm[0]:<5}{cm[1]:<5}{cm[2]:<5}{cm[3]:<5} {self.thresh} {ACC*100:.2f}" \
				f" {PPV*100:.2f} {TPR*100:.2f} {TNR*100:.2f} {F1_score*100:.2f}"
		print(pnts3)
		return pnts3

	def get_ground_truth(self, pn_dir):
		with open(f'{pn_dir}/positive.txt', 'r') as f:
			positive = f.readlines()
		with open(f'{pn_dir}/negative.txt', 'r') as f:
			negative = f.readlines()
		positive = [x.strip('\n') for x in positive]
		negative = [x.strip('\n') for x in negative]
		p_and_n = positive + negative
		gt = [[x, 0] for x in positive]
		gt_data = np.array(gt)
		return gt_data

	def get_detect_data(self, need_infer=False, weight_path=None, v5_out=None):
		"""
		:param need_infer:
		:param weight_path:
		:param v5_out:
		:return: dt_data, an array, shape(nfiles, 50, 2)
				nfiles: number of output files, 100 represents a maximum of 100 targets in an image.
				2 represents class and confidence.
				dt_data[dt_data[:, :, 1] != -2]，筛选有目标的jpg,filename会重复。shape, (n_detected_files, 3) 检测到的目标的图片数目。
		"""
		assert v5_out is not None
		assert not need_infer
		files = os.listdir(v5_out)
		imgs = sorted([tmp.strip('\n') for tmp in files if tmp.endswith(self.img_suffix)])
		txts = [tmp.strip('\n') for tmp in files if tmp.endswith('.txt')]

		dt_data = np.zeros((len(imgs), 100, 7)).astype(np.object)  # 7: img cls conf bbox(xcycwh_percent)
		dt_data[:, :, :] = -2
		cls_c = np.zeros(100).astype(np.int)  # 分类count
		for i, img in enumerate(imgs):
			file_stem = Path(img).stem
			txt_file = f'{v5_out}/{file_stem}.txt'
			if os.path.exists(txt_file):
				with open(txt_file, 'r') as f:
					txt_data = f.readlines()
				# if len(txt_data) > 1:
				# 	print(img)
				for j, t in enumerate(txt_data):  #
					cls, conf, xc, yc, w, h = t.split()
					# tmp = [img, int(cls), float(conf)]
					tmp = [img, int(cls), float(conf), float(xc), float(yc), float(w), float(h)]
					dt_data[i, j, :] = np.array(tmp)
					cls_c[int(cls)] += 1
			else:
				dt_data[i, 0, :] = np.array([img, -2, -2, -2, -2, -2, -2])

		cls_c = cls_c[cls_c[:] != 0]
		print(
			f'detected output: {len(imgs)} imgs[.{self.img_suffix}], {len(txts)} txts. {np.sum(cls_c)} detected targets.')
		prnts = '\n'.join([f'class{i}: {count} targets detected' for i, count in enumerate(cls_c)])
		print(prnts)
		return dt_data

	def detect(self):
		self.net.eval()
		os.system(r'rm -rf detect_ret/detect.txt')
		for i, filename in enumerate(self.cfg.filenames):
			t = time.time()
			img_path = f"{self.cfg.infer_img_dir}/{filename}"
			img = Image.open(img_path)
			img, outs = self.infer(img)
			# img.save(f'/Users/tanmenglu/Downloads/AutoDrive/ret/{filename}')
			print(f"{filename} time: {time.time()-t:.2f} FPS: {1/(time.time()-t):.2f}")
			strs = ''
			for ot in outs:
				strs += ' ' + ','.join(ot)
			os.system(f'echo "{filename}{strs}" >>detect_ret/detect.txt')


def plot():
	import matplotlib.pyplot as plt
	threshes = [i / 100 for i in range(10, 91, 2)]
	xs = []
	ys = []
	for thresh in threshes:
		ret = GCM(thresh=thresh, v5_out=v5_out)
		print(ret)
		ret = ret.split('\n')[-2].split()
		xs.append((ret[1]))
		ys.append([ret[3].strip('%'), ret[5].strip('%'), ret[7].strip('%'), ret[9].strip('%'), ret[11].strip('%')])

	xs = np.array(xs).astype(np.float).reshape((-1, 1))
	xs = np.repeat(xs, 5, axis=1)
	ys = np.array(ys).astype(np.float)
	print(xs.shape, ys.shape)
	print(xs, ys)
	lbs = ['ACC', 'PPV', 'RECALL', 'TNR', 'F1_score']
	colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(xs.shape[1])]
	colors = [[tmp/255 for tmp in tmp2] for tmp2 in colors]
	for i in range(xs.shape[1]):
		x = xs[:, i]
		y = ys[:, i]
		print(colors[i])
		plt.plot(x, y, label=lbs[i], color=colors[i])

	# 标出max：
	ys_max = np.max(ys, axis=0)  # 每一列是一个种类，对每一列求最大值m
	ys_max_i = np.argmax(ys, axis=0)  # [c1,c2,c3,c4,c5]
	xs_max = xs[:, 0][ys_max_i]
	print(xs_max, ys_max_i, ys_max)
	for i in range(len(xs_max)):
		x = xs_max[i]
		y = ys_max[i]
		plt.text(x, y, f"{lbs[i]} {x} {y}", color=colors[i])
		plt.plot(x, y, 'o', color=colors[i])

	plt.xticks()
	plt.legend(loc='best')
	plt.xlabel('thresh')
	plt.ylabel('precent/%')
	plt.savefig('xxx.png', bbox_inches='tight')


if __name__ == "__main__":
	GCM = GetConfusingMatrix()

	path = "/home/zzd/datasets/crosswalk/testsets_1770"
	ret_all = []
	head = f"trs tes {'ALL':<5}{'TP':<5}{'FP':<5}{'FN':<5}{'TN':<5} thr acc   P     R     TNR   F1    {'time':<7} fps\n"
	ret_all.append(head)
	train_sizes = [640]
	test_sizes = range(128, 160, 32)
	for trs in train_sizes:  # train_size
		for tes in test_sizes:  # test_size
			v5_out = f"{path}/dense_ts_roi_SE300epoch_{trs}_{tes}_output"
			v5_out = f'{path}/SEv5m_300EP_no_roi'
			ret = GCM.analyse(thresh=0.5, v5_out=v5_out, pn_dir=path)
			# 读取inference时间
			with open(f'{v5_out}/detect.log', 'r') as f:
				log = f.readlines()
			time, fps = re.findall(f'\d+.\d+', log[-1])
			time = time[:-1]
			fps = fps[:-3]
			ret_all.append(f'{trs} {tes} {ret} {time} {fps}\n')
	with open('confusing_matrix_dense_roi_SE300epoch.txt', 'w') as f:
		f.writelines(ret_all)
