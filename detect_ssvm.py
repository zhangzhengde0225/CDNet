"""
detect.py是检测测试集的。
这个是检测origin_videos视频split的，用于评估斑马线过线行为检测的结果。
"""
import argparse
import yaml
import os
import detect
import post


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights', nargs='+', type=str, default='v5m2cls.pt',
						help='model.pt path(s)')
	parser.add_argument('--source', type=str, default='example/images', help='source')  # file/folder, 0 for webcam
	parser.add_argument('--output', type=str, default='example/output', help='output folder')  # output folder
	parser.add_argument('--img-size', type=int, default=288, help='inference size (pixels)')
	parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
	parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
	parser.add_argument('--device', default='1,2', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
	parser.add_argument('--view-img', action='store_true', help='display results')
	parser.add_argument('--save-txt', type=bool, default=True, help='save results to *.txt')
	parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
	parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
	parser.add_argument('--augment', action='store_true', help='augmented inference')
	parser.add_argument('--update', action='store_true', help='update all models')
	parser.add_argument('--control_line_setting', type=str, default='detect/cl_setting.yaml', help='control line setting')
	parser.add_argument('--select_control_line', type=str, default='general', help='select which control line. i.e. general, 0036')
	parser.add_argument('--field_size', type=int, default=6, help='receptive filed size for post')
	parser.add_argument('--plot_classes', type=list, default=['crosswalk'], help='specifies which classes will be drawn')
	parser.add_argument('--use_roi', type=bool, default=False, help='use roi to accelerate inference speed')

	opt = parser.parse_args()

	print(opt)
	# opt.weights = '/Users/tanmenglu/weights/v5mCD640.pt'
	# for size in range(96, 833, 32):
	# 	opt.img_size = size
	# 	run()
	# exit()
	# ---- shnavi ---- #
	origin_videos_path = '/root/datasets/crosswalk/origin_videos'
	# 读取yaml
	with open(f'{origin_videos_path}/videos_categories.yaml', 'r') as f:
		data = yaml.load(f, Loader=yaml.FullLoader)
	data = dict(zip([k for k in data.keys()], [v.split(',') for v in data.values()]))  # 把字典的只用逗号隔开变成列表
	img_folds = sum([x for x in data.values()], [])  # 只取值并把嵌套的列表展开

	for roiii in [True, False]:
		bnxx = 'bNSRS' if roiii else 'bNSS'
		opt.use_roi = True
		for fd in img_folds:
			opt.source = f'{origin_videos_path}/{fd}'
			base_path = "/root/PycharmProject/book_pytorch/yolov5_zzd/runs"
			# exp = "v5mCD640exp"
			exp = "v5mCD640SE300epochexp"
			opt.weights = f"{base_path}/{exp}/weights/best.pt"
			test_sizes = [288, 640]
			opt.select_control_line = fd
			for tes in test_sizes:

				opp = f"{origin_videos_path}/outputs_{bnxx}/ROI_SSVM_{fd}_{tes}"
				if os.path.exists(opp):
					print(f'{opp} exists, pass')
					continue
				opt.output = opp
				opt.img_size = tes
				print(f"exp: {exp} test size: {tes} fold: {fd} roi: {opt.use_roi}")
				dp = post.DmPost(opt)
				detect.run(opt, dp)

