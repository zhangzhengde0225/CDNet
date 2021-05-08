"""
绘制result的图
"""

import os
import matplotlib.pyplot as plt
import yaml
import numpy as np
import matplotlib

matplotlib.rcParams['xtick.direction'] = 'in'  # 刻度线向内
matplotlib.rcParams['ytick.direction'] = 'in'
c = [
	'lightcoral', 'limegreen', 'royalblue', 'orange', 'mediumorchid',
	'orangered', 'lawngreen', 'deepskyblue', 'gold', 'fuchsia',
	'sandybrown', 'cyan']
c = [
	'lightcoral', 'limegreen', 'royalblue', 'orange', 'mediumorchid',
	'deepskyblue', 'gold', 'fuchsia',
	'sandybrown', 'cyan']
c = [
	'black', 'lightcoral', 'limegreen', 'royalblue', 'orange', 'mediumorchid',
	'deepskyblue', '#EE3B3B', 'gold', 'fuchsia',
	'sandybrown', 'cyan']


class CDNetPlot(object):
	def __init__(self):
		self.data = self.load_data('data/results_2.yaml')

	def load_data(self, path):
		with open(path, 'r') as f:
			data = yaml.load(f, Loader=yaml.FullLoader)
		return data

	def plot(self):
		d = self.data
		for i, k in enumerate(d.keys()):
			v = [float(x) for x in d[k].split()]
			print(k, v, type(v))
			x = np.array([v[0], v[2]]).astype(np.float32)
			y = np.array([v[1], v[3]]).astype(np.float32)
			# 划线
			lb = '$\mathbf{'+k+'}$'
			plt.plot(
				x, y, ls='-', label=lb, c=c[i], linewidth=3,
				marker='', markersize=10)
			# 画点288
			plt.plot(v[0], v[1], marker='^', c=c[i], markersize=15)
			plt.plot(v[2], v[3], marker='s', c=c[i], markersize=15)
			# 添加额外的标签
			if False:
				if i == 0:
					plt.text(v[0]-12, v[1]-1.5, '288', size=16)
					plt.text(v[2] - 12, v[3] - 1.5, '640', size=16)

		fontdict = {'size': 18, 'weight': 'bold'}
		plt.legend(loc=2, bbox_to_anchor=(1.0, 1.0), borderaxespad=0.)
		plt.grid()
		plt.xlabel('Detection Speed (FPS)', fontdict=fontdict)
		plt.ylabel('F1 score (%)', fontdict=fontdict)
		plt.xticks(range(0, 100, 5), size=12, weight='bold')
		plt.xlim(1, 42)
		plt.yticks(range(60, 101, 5), size=12, weight='bold')
		plt.ylim(77, 98)
		# plt.xticks(range(0, 300, 50))
		plt.show()
		# plt.savefig('results.png', dpi=2000, bbox_inches='tight')


if __name__ == '__main__':
	cdnetp = CDNetPlot()
	cdnetp.plot()
