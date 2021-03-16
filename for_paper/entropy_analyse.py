"""
分析使用NST前后的数据集，根据信息论，从信息熵出发分析性能。
"""
import os
import numpy as np


class EntropyAnalyse(object):
	def __init__(self):
		self.lb_2cls = "/home/zzd/datasets/crosswalk/train_data_v5_format/labels/train"
		self.lb_1cls = "/home/zzd/datasets/crosswalk/train_data_v5_format_1cls/labels/train"

		self.out_1cls_288 = "/home/zzd/datasets/crosswalk/testsets_1770/dense_testsize_full_1cls_640_288_output"
		self.out_1cls_640 = "/home/zzd/datasets/crosswalk/testsets_1770/dense_testsize_full_1cls_640_640_output"
		self.lb_2cls_te = "/home/zzd/datasets/crosswalk/train_data_v5_format/labels/test"

	def run(self):

		e = self.get_error_rate(out_path=self.out_1cls_640)
		# print(e)
		# exit()
		# 第一步，读取概率
		P2cls = self.get_P_from_label(path=self.lb_2cls)
		H2 = self.cal_information_entropy(P2cls)
		print(H2)
		exit()
		P1cls = self.get_P_from_label(path=self.lb_1cls)


		# P = np.array([0.1, 0.9, 0])
		H1 = self.cal_information_entropy(P1cls)  # 熵

		print(H1, H2)

		# 统计在1cls中误检为crosswalk的数目，分288和640
		


	def get_error_rate(self, out_path):
		"""
		根据2cls的测试集的标注文件和1cls时的检测结果，统计在所有标注的guide_arrows中，被误检为crosswalk的错误数
		:return:
		"""
		lb2te = self.lb_2cls_te
		txts = [x for x in os.listdir(lb2te) if x.endswith('.txt')]
		error = np.zeros(3)  # 总标注数目，标注了1个同时检测为crosswalk，标注2个同时检测出2个crosswalk
		for i, txt in enumerate(txts):
			# print(txt)
			classes = self.read_txt(f'{lb2te}/{txt}')
			if '1' in classes:
				classes_ga = [x for x in classes if x == '1']  # 只有guide arrows
				both = len(classes) != len(classes_ga)  # both为真时，表示该标注文件同时标注了crosswalk和guidearrows
				assert len(classes_ga) == 1
				classes2 = self.read_txt(f'{out_path}/{txt}')
				classes2 = [] if classes2 is None else classes2
				# print(classes, classes2, both)
				error[0] += 1

				if not both and '0' in classes2:
					error[1] += 1
				if both and len(classes2) >= 2:
					print(f'{txt} 标注了2个，且有GA，检测出大于2个 {classes} {classes2}')
					error[1] += 1
				print(f'[{i+1:>2}/{len(txts)}] {error} {out_path}/{txt}')

		return 0


	def get_P_from_label(self, path):
		txts = [x for x in os.listdir(path) if x.endswith('.txt')]
		assert len(txts) != 0
		"""
		根据yolo的目标检测算法，每张图的样本总数应该是：7x7x3
		"""
		P = np.zeros(3)  # crosswalk guide arrows total
		for i, txt in enumerate(txts):
			with open(f'{path}/{txt}', 'r') as f:
				data = f.readlines()
			classes = [x.split()[0] for x in data]  # 标注文件中的类别
			assert len(classes) != 0
			for j, cls in enumerate(classes):
				if cls == '0':
					P[0] += 1
				elif cls == '1':
					P[1] += 1
					if len(classes) >=2:
						print(f'同时存在，{classes}')
				else:
					raise NameError(f'标注文件读取到的分类{cls}不正确，文件：{path}/{txt}')
			P[2] += (7*7*3 - len(classes))

			print(f'\r[{i+1:>2}/{len(txts)}] {P} ', end='')

		P = P/np.sum(P)
		print(P)
		return P

	def cal_information_entropy(self, P):
		"""
		输入的P是一个ndarray，长度是可能的符号数目，每个值是对应的概率Pxi
		H(x) = - sum(Pxi * log(Pxi))
		:param P:
		:return:
		"""
		P = P[P != 0]  # 筛选掉为0的项
		return -np.sum(np.multiply(P, np.log2(P)))

	def read_txt(self, txt):
		if os.path.exists(txt):
			with open(txt, 'r') as f:
				data = f.readlines()
			return [x.split()[0] for x in data]
		else:
			return None


if __name__ == '__main__':
	ea = EntropyAnalyse()
	ea.run()
