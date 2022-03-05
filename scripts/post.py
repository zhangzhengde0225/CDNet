"""
post for crosswalk detection, the vector crossing method used to detect the vehicle crossing crosswalk behavior
"""

import os
import cv2
import numpy as np
import yaml
import copy
from pathlib import Path
import re

__author__ = 'drivener@163.com'
__date__ = '20200929'


class DmPost(object):
	"""damei post"""
	def __init__(self, opt):
		self.opt = opt
		self.crossout = self.init_crossout()
		self.vetors = self.init_vectors()
		self.old_vectors, self.frame_thresh, self.old_fileorder = self.init_jitter_pretection_params()  # 用于抖动保护
		self.logo = self.init_logo()
		self.rect = self.init_rect()
		self.t_inter = self.init_others()
		self.cl, self.rl = self.init_clrl()

	def init_crossout(self):
		crossout = np.zeros(
			(self.opt.field_size, 8))
		# 5 rows. 8 columns ：out_index, detect_id, fileorder, is crosswalk exists, xc, yc, count，recording_flag
		crossout[:, :] = -2  # -2 represents undetected
		crossout[:, 0] = range(len(crossout))  # init out_index
		crossout[:, 6:8] = 0  # init count and recording_flag to 0
		return crossout

	def init_vectors(self):
		vector_size = 600 * 30  # maximun 600 seconds, sampling ratio 30。
		vectors = np.zeros((vector_size, 2))  # 2: store xc, yc
		vectors[:, :] = -2  # init
		return vectors

	def init_jitter_pretection_params(self, time_thresh=2):
		raw_video_sampling_rate = 25  # 原始的视频的采样率，1帧对应于1/25 = 40ms
		video2imgs_sampling_interval = 5  # 把原始视频转换为视频的采样间隔，如果为1, 全采样，1帧还是对应1/25，如果为5，1帧对应于1/(25/5) = 200ms
		ft = int(time_thresh/(video2imgs_sampling_interval/raw_video_sampling_rate))  # frame thresh, 为10.
		return None, ft, -ft

	def init_logo(self):
		logo_path = "data/SHNAVI.jpg"
		logo = cv2.imread(logo_path)
		imgs = [x for x in os.listdir(self.opt.source) if x.endswith('jpg') or x.endswith('png')]
		if len(imgs) == 0:
			raise NameError(f'did not get imgs from {self.opt.source}, support suffix: .jpg .png')
		img_size = cv2.imread(f'{self.opt.source}/{imgs[0]}').shape[:2]
		logo_size = logo.shape[:2]  # 248, 1217
		scale_factor = 3.5
		resized_logo_size = (
		int(img_size[1] / scale_factor), int(img_size[1] / scale_factor * logo_size[0] / logo_size[1]))
		logo = cv2.resize(logo, resized_logo_size, interpolation=cv2.INTER_LINEAR)
		return logo

	def init_rect(self):
		rect = cv2.imread("settings/rect.jpg")
		rect_resized_size = (self.logo.shape[1], 200)
		rect = cv2.resize(rect, rect_resized_size, interpolation=cv2.INTER_LINEAR)
		return rect

	def init_clrl(self):
		# get opt
		opt = self.opt
		with open(opt.control_line_setting, 'r') as f:
			conls = yaml.load(f, Loader=yaml.FullLoader)
		# print(conls)
		cl = conls[opt.select_control_line]  # control line
		rl = conls['red_line']  # red_line
		return cl, rl

	def init_others(self):
		FPS = 30  # 1s 有25张图
		sampling_ratio = 30  # 每4张采样1张
		sampling_rate = FPS / sampling_ratio  # 采样率,1秒5张
		base_time = 0  # 基准时间，检测的图片超过chunk大小时，基准时间要增加
		t_inter = 1 / sampling_rate  # time interpolation 每张0.2秒
		return t_inter

	def dmpost(self, img, det, det_id, filename, names, cls='crosswalk'):
		"""
		:param img:
		:param det: None或tensor，shape:(n, 6), n是n个目标，6是已经scale到原图的x1,y1,x2,y2和confidence、class
		:param filename: filename0001.jpg获取order，或者输入0036_1，0036是视频名，1是第1帧。
		:param names: 模型中的类名
		:param cls: 后处理筛选的分类，默认是crosswalk
		:return:
		"""
		opt = self.opt
		crossout = self.crossout
		vectors = self.vetors

		# 0000 0001 0002 0003 and so on # 1 2 3 4 ....
		try:
			fs = Path(filename).stem.split('_')[-1]
			fileorder = int(re.findall('(\d+)', fs)[-1])
		except Exception as e:
			raise NameError(
				f'Filename error: {filename}. Please make sure that the image name contains the frame order, '
				f'which can be separated by any letter or "_". '
				f'Such as: 00001.jpg, 0001.jpg, filename_0001.jpg, filename0001.jpg, xx_filename0001.jpg')

		# 绘制上下控制线
		# cl = opt.control_line  # control line, [360, 700]
		cl = self.cl
		rl = self.rl
		lt = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
		tf = max(lt - 1, 1)  # 2
		h, w = img.shape[:2]
		top = [(0, cl[0]), (w, cl[0])]
		button = [(0, cl[1]), (w, cl[1])]
		middle = [(int(w / 4), int(cl[0]+(cl[1]-cl[0])*rl)), (int(3 * w / 4), int(cl[0]+(cl[1]-cl[0])*rl))]
		arrow = [(int(w / 2), middle[0][1]), (int(w / 2), middle[0][1] + 50)]
		textpos = (top[0][0] + 10, top[0][1] + 30)

		cv2.line(img, top[0], top[1], color=(30, 30, 224), thickness=lt)
		cv2.line(img, button[0], button[1], color=(30, 30, 224), thickness=lt)
		cv2.line(img, middle[0], middle[1], color=(120, 85, 220), thickness=lt)  # 粉色
		cv2.arrowedLine(img, arrow[0], arrow[1], (120, 85, 220), 5, 8, 0, 0.3)
		cv2.putText(img, 'Control Line', textpos, 0, lt / 3, (30, 30, 224), thickness=tf, lineType=cv2.LINE_AA)

		cls_idx = names.index(cls)  # class index i.e. 0 for crosswalk
		det_include_class = cls_idx in det[:, 5] if det is not None else False
		if det is not None and det_include_class:
			det = det.cpu().numpy()
			crosswalk = np.zeros((det[det[:, 5] == 0].shape[0], 8))
			# [n, 6]变成[n, 8]，去掉cls。8分别是x1,y1,x2,y2,conf,cx,cy,is_in_control_line
			# print(det.shape, det[:, 5] == 1)
			crosswalk[:, :5] = det[det[:, 5] == 0][:, :5]  # 筛选只要crosswalk，默认起序号为0, 0 1 2 3 4
			cx = (crosswalk[:, 0] + crosswalk[:, 2]) // 2
			cy = (crosswalk[:, 1] + crosswalk[:, 3]) // 2
			crosswalk[:, 5] = cx
			crosswalk[:, 6] = cy
			is_in_cl = (crosswalk[:, 6] > cl[0]) | (crosswalk[:, 6] < cl[1])
			crosswalk[:, 7] = is_in_cl
			if crosswalk.shape[0] > 1:
				# 同时检测到多条斑马线，根据之前记录的crossout，确定选择哪一条
				# lastxc, lastyc = crossout
				last_co = crossout[crossout[:, 3] != -2]  # last crossout
				# print(last_co, last_co.shape)
				if len(last_co) == 0:  # 都是空的，，那就使用第一条
					valid_idx = 0
				else:  # 计算检测到的斑马线的中心与上次记录的中心哪个接近就用哪个
					lastcxy = last_co[-1][4:6]
					currentcxy = crosswalk[:, 5:7]
					# print(lastcxy, lastcxy.shape, currentcxy, currentcxy.shape)
					distances = np.sum(np.square(currentcxy - lastcxy), axis=1)  # 距离
					valid_idx = np.argmin(distances)
				# print(f'WARNING: detect.py post, the detected crosswalk is more than one, use the {valid_idx+1} one')
				crosswalk = crosswalk[valid_idx, :].reshape(1, -1)
				det = det[valid_idx, :].reshape(1, -1)
		else:
			crosswalk = np.zeros((1, 8))

		# print(crosswalk.shape)
		# print(det.shape)

		if det_id < crossout.shape[0]:  # detected_img id < 5
			# 该列更新值 n列：out_index, detect_id, fileorder, 有无斑马线, xc, yc
			crossout[det_id, 1] = det_id
			crossout[det_id, 2] = fileorder
			crossout[det_id, 3] = crosswalk[0, 7]
			crossout[det_id, 4:6] = crosswalk[0, 5: 7]  # xc, yc
			index = det_id
		else:
			crossout[0:-1:, 1::] = crossout[1::, 1::]  # 除了序号列的所有行向上平移一格
			# 最后一列更新值
			crossout[-1, 1] = det_id
			crossout[-1, 2] = fileorder
			crossout[-1, 3] = crosswalk[0, 7]
			crossout[-1, 4:6] = crosswalk[0, 5: 7]  # xc, yc
			index = len(crossout) - 1

		# print(crossout[:11, :])
		exist, vector, scale = self.decode_crossout(crossout, index)
		recording = crossout[index, 7]

		if recording == 1 and vector is not None:
			vectors[opt.v_idx, :] = vector[0]  # 写入矢量的第一个点
			vectors[opt.v_idx + 1, :] = vector[1]  # 写入矢量的第二个点
			opt.v_idx += 1
		elif recording == 1 and vector is None:  # 记录但没有矢量传入，保持原来的
			pass
		else:
			if vectors[0, 0] != -2:
				self.old_vectors = copy.deepcopy(vectors)  # 初始化前保存到另一个变量中
			vectors[:, :] = -2  # 再次初始化
			opt.v_idx = 0

		speed = None if scale is None else float((vector[1][1] - vector[0][1]) / (self.t_inter * scale))
		# 向量的y距离除以scale再除以time interpolate

		# 绘制logo和结果
		# logo_pos = (int(img.shape[1]/2-logo.shape[1]/2), cl[1]-logo.shape[0])
		logo_pos = (20, 20)  # w h
		img = self.imgAdd(self.logo, img, logo_pos[1], logo_pos[0], alpha=0.5)
		rect_pos = (20, 20 + 5 + self.logo.shape[0])
		img = self.imgAdd(self.rect, img, rect_pos[1], rect_pos[0], alpha=0.5)

		pos = (20 + 20, 20 + 5 + self.logo.shape[0] + 20 + 30)
		self.imgputText(img, f'crosswalk: {exist}', pos, lt, tf)
		pos = (20 + 20, 20 + 5 + self.logo.shape[0] + 20 + 30 + 40)
		self.imgputText(img, f'speed: {speed:.2f}', pos, lt, tf) if speed is not None else None
		self.imgputText(img, f'speed: {speed}', pos, lt, tf) if speed is None else None

		vt = vectors[vectors[:, 0] != -2].astype(np.int)  # 筛选t
		self.imgplotVectors(img, vt) if vector is not None else None

		# 计数，vectors有值、向量经过中间线、当前vector不为None时，count+1
		crossout[index, 6] = crossout[index - 1, 6]  # 先同步count与前一个相同
		# print(self.old_fileorder, int(crossout[index, 2]))
		if vt.shape[0] != 0:  # 非空就有可能过线，空不可能过线。
			# print(vt.shape)  # (n, 2), n是矢量的n个点，2是像素坐标x, y
			if vt[0, 1] < np.mean(cl):  # 起点在控制线上方才行
				intersect = vt[vt[:, 1] > middle[0][1]]  # 所有y坐标大于cl的点
				# 仅使用交叉数目等于1有个问题，如果前一帧刚刚交叉，后一帧抖动回去，又会计入一次，添加条件inter2
				intersect2 = vt[-1, 1] > middle[0][1]  # 最后一个交叉点
				# print('inter', intersect)
				# print(vt, intersect, intersect2)
				if intersect.shape[0] == 1 and vector is not None and intersect2:
					# 添加抖动保护机制，本次计数的帧与上一次相比小于10帧时(对应于2s)，把当前的vectors与上一个vectors组合起来，
					c_fileorder = int(crossout[index, 2])
					if (c_fileorder - self.old_fileorder) < self.frame_thresh:  # 10
						# print('\nxxx', c_fileorder, self.old_fileorder, self.frame_thresh)
						vto = self.old_vectors[self.old_vectors[:, 0] != -2].astype(np.int)
						new_vt = np.concatenate((vto, vt), axis=0)
						self.imgplotVectors(img, new_vt)
						vectors[:new_vt.shape[0], :] = new_vt
					else:
						crossout[index, 6] += 1  # count+1
						prt_str = \
							f'\nThe vehicle crossed a crosswalk in {filename}!! count+1, conf: {crosswalk[0, 4]:.2f}, ' \
							f'current count: {int(crossout[index, 6])}.'
						# print(prt_str)
						os.system(f'echo "{prt_str}" >> {opt.output}/detect.log')
						self.old_fileorder = copy.deepcopy(c_fileorder)
		count = int(crossout[index, 6])

		pos = (20 + 20, 20 + 5 + self.logo.shape[0] + 20 + 30 + 40 + 40)
		self.imgputText(img, f'count: {count}', pos, lt, tf)

		# 打印状态：有vectors时就是crossing
		pos = (20 + 20, 20 + 5 + self.logo.shape[0] + 20 + 30 + 40 + 40 + 40)
		status = 'No crosswalk' if vt.shape[0] == 0 else 'Crossing'
		self.imgputText(img, f'status: {status}', pos, lt, tf)

		prt_str = f'{filename} detect_id: {det_id} speed: {speed} count: {count} status: {status}'
		# print(prt_str)
		os.system(f'echo "{prt_str}" >> {opt.output}/detect.log')

		return img

	def decode_crossout(self, crossout, index, vector_threash=20, vector_max_threash=600):
		"""
		解码crossout, 输出当前图是否有斑马线，斑马线位移矢量，时间尺度（索引的间距）
		count计数算法：该矢量经过中心线时，计数+1。
		:param crossout:
		:index: 当前位置索引
		:recursive_bits: 向前递归位数
		:vector_threash: 向量的模的阈值，大于该阈值且y的变化为负数时，设置recording为0，断开count。
		:return:
		"""
		exist = crossout[index, 3]
		co = crossout[crossout[:, 3] == 1]  # 所有存在的行
		if exist == 0:
			if co.shape[0] == 0:  # 感受野范围内都没有斑马线
				crossout[:, 7] = 0  # recording置0
			return False, None, None
		else:
			if co.shape[0] == 1:  # 只有最后一行有
				crossout[:, 7] = 0  # recording置0
				return False, None, None
			else:
				scale = co[-1, 1] - co[-2, 1]  # detected_id的差
				vector = [co[-2, 4:6], co[-1, 4:6]]  # 2个点 ((xc1, yc1), (xc2, yc2))
				vector2 = vector[1] - vector[0]  # (x2-x1) (y2-y1)
				length = np.sqrt(vector2[0] ** 2 + vector2[1] ** 2)
				y_shift = vector2[1]
				# print(vector2, vector2/1280, vector2/720, length, y_shift)
				if length > vector_threash and y_shift < 0:
					crossout[:, 7] = 0  # recording置0
				elif length > vector_max_threash:  # 有时候会出现超级长的大于300像素，筛掉，约((680-400)*2/3)**2 680和400是控制线
					crossout[:, 7] = 0
				else:
					crossout[:, 7] = 1  # recording置1
				# print(crossout[:, 7], length)
				return True, vector, scale

	def imgAdd(self, small_img, big_image, x, y, alpha=0.5):
		"""
		把小图贴到大图的xy位置，透明度设置为0.5
		"""
		row, col = small_img.shape[:2]
		if small_img.shape[0] > big_image.shape[0] or small_img.shape[1] > big_image.shape[1]:
			raise NameError(f'imgAdd, the size of small img bigger than big img.')
		roi = big_image[x:x + row, y:y + col, :]
		roi = cv2.addWeighted(small_img, alpha, roi, 1 - alpha, 0)
		big_image[x:x + row, y:y + col] = roi
		return big_image

	def imgputText(self, img, txt, pos, lt, tf):
		cv2.putText(img, txt, pos, 0, lt / 3, (30, 30, 224), thickness=tf, lineType=cv2.LINE_AA)

	def imgplotVectors(self, img, vt):
		if vt.shape[0] == 0:
			return
		for i in range(vt.shape[0] - 1):
			pt1, pt2 = tuple(vt[i]), tuple(vt[i + 1])
			cv2.arrowedLine(img, pt1, pt2, (0, 0, 255), 5, 8, 0, 0.3)
