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


class CDNetPlot(object):
	def __init__(self):
		self.data = self.load_data('data/results.yaml')

	def load_data(self, path):
		with open(path, 'r') as f:
			data = yaml.load(f, Loader=yaml.FullLoader)
		return data

	def plot(self):
		d = self.data
		for i, k in enumerate(d.keys()):
			v = [float(x) for x in d[k].split()]
			print(k, v, type(v))
			x = np.array([v[0], v[2]]).astype(np.float)
			y = np.array([v[1], v[3]]).astype(np.float)
			plt.plot(
				x, y, ls='-', label=k, c=c[i], linewidth=3,
				marker='o', markersize=10)
			# 添加额外的标签
			if True:
				if i == 0:
					plt.text(v[0]-12, v[1]-1.5, '288', size=16)
					plt.text(v[2] - 12, v[3] - 1.5, '640', size=16)

		fontdict = {'size': 18, 'weight': 'bold'}
		plt.legend(loc='best')
		plt.grid()
		plt.xlabel('CPU Speed (ms/img)', fontdict=fontdict)
		plt.ylabel('F1$\mathbf{_{score}}$ (%)', fontdict=fontdict)
		plt.xticks(range(0, 701, 100), size=12, weight='bold')
		plt.xlim(50, 700)
		plt.yticks(range(75, 101, 5), size=12, weight='bold')
		plt.ylim(75, 98)
		# plt.xticks(range(0, 300, 50))
		# plt.show()
		plt.savefig('results.png', dip=300, bbox_inches='tight')


if __name__ == '__main__':
	cdnetp = CDNetPlot()
	cdnetp.plot()
