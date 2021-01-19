"""
convert xxx.pt to xxx.wts

need install damei library:
	pip install damei -i https://pypi.Python.org/simple
"""
import damei as dm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
	'weights', type=str, default='runs/exp0/weights/best.pt',
	help='path to weights [xx.pt]')
parser.add_argument('--output_dir', type=str, default=None)
opt = parser.parse_args()

dm.post.pt2wts(opt.weights, opt.output_dir)

