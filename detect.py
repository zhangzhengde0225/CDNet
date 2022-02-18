import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import numpy as np
import yaml

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from scripts import post


def detect(opt, dp, save_img=False):
	out, source, weights, view_img, save_txt, imgsz = \
		opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
	webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

	# Initialize
	device = select_device(opt.device)
	if os.path.exists(out):
		shutil.rmtree(out)  # delete output folder
	os.makedirs(out)  # make new output folder
	half = device.type != 'cpu'  # half precision only supported on CUDA

	# Load model
	model = attempt_load(weights, map_location=device)  # load FP32 model

	imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size 如果不是32的倍数，就向上取整调整至32的倍数并答应warning

	if half:
		model.half()  # to FP16

	# Second-stage classifier
	classify = False
	if classify:
		modelc = load_classifier(name='resnet101', n=2)  # initialize
		modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
		modelc.to(device).eval()

	# Set Dataloader
	vid_path, vid_writer = None, None
	if opt.use_roi:
		# print(dp.cl)
		# print(dp.cl[0], dp.cl[1])
		# cl = opt.control_line
		cl = dp.cl
		roi_in_pixels = np.array([0, cl[0], 1280, cl[1]])  # two points coor, x1, y1, x2, y2
	else:
		roi_in_pixels = None

	if webcam:
		view_img = True
		cudnn.benchmark = True  # set True to speed up constant image size inference
		dataset = LoadStreams(source, img_size=imgsz)
	else:
		save_img = True
		dataset = LoadImages(source, img_size=imgsz, roi=roi_in_pixels)

	# Get names and colors
	names = model.module.names if hasattr(model, 'module') else model.names  # 解决GPU保存的模型多了module属性的问题
	colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]  # 随机颜色，对应names，names是class

	# fix issue: when single cls, names = ['item'] rather than names = ['crosswalk']
	if 'item' in names:
		names = ['crosswalk']

	# prune
	# torch_utils.prune(model, 0.7)
	model.eval()

	# Run inference
	t0 = time.time()
	img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
	_ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once 空跑一次，释放！！牛逼

	detected_img_id = 0
	time_list = [None] * len(dataset)
	bar = tqdm(dataset)
	for iii, (path, img, im0s, vid_cap, recover) in enumerate(bar):
		# print(img.shape, im0s.shape, vid_cap)
		# exit()

		# img.shape [3, 384, 640] im0s.shape [720, 1280, 3] None
		img = torch.from_numpy(img).to(device)
		img = img.half() if half else img.float()  # uint8 to fp16/32
		img /= 255.0  # 0 - 255 to 0.0 - 1.0
		if img.ndimension() == 3:
			img = img.unsqueeze(0)  # 从[3, h, w]转换为[batch_size, 3, h, w]的形式

		# Inference
		t1 = time_synchronized()
		# print('aug', opt.augment)  # False
		pred = model(img, augment=opt.augment)[0]
		# print(pred.shape) [1, 15120, 25]
		# Apply NMS
		pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
		t2 = time_synchronized()
		infer_time = t2 - t1
		time_list[iii] = t2-t1

		# print('\n', len(pred), pred, recover)  # list 长度是bs，代表每张图, 元素tensor，代表检测到的目标，每个tensor.shape [n, 6] xy4, conf, cls

		# Apply Classifier
		if classify:
			pred = apply_classifier(pred, modelc, img, im0s)

		# Process detections
		for i, det in enumerate(pred):  # detections per image
			if opt.use_roi and det is not None:
				small_img_shape = torch.from_numpy(np.array([recover[1], recover[0]]).astype(np.float))
				det[:, 0], det[:, 2] = det[:, 0] + recover[2], det[:, 2] + recover[2]
				det[:, 1], det[:, 3] = det[:, 1] + recover[3], det[:, 3] + recover[3]
			else:
				small_img_shape = img.shape[2::]
			if webcam:  # batch_size >= 1
				p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
			else:
				p, s, im0 = path, '', im0s  # im0s是原图

			save_path = str(Path(out) / Path(p).name)  # output/filenamexxxx.jpg
			txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
			# output/filenamexxxx.txt
			s += '%gx%g ' % img.shape[2:]  # print string, 640x640
			gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
			# 本来是[720, 1280, 3]，重复取，变成[1280, 720, 1280, 720]
			if det is not None and len(det):
				# Rescale boxes from img_size to im0 size
				det[:, :4] = scale_coords(small_img_shape, det[:, :4], im0.shape).round()  # 转换成原图的x1 y1 x2 y1，像素值
				# Print results
				for c in det[:, -1].unique():
					n = (det[:, -1] == c).sum()  # detections per class
					s += '%g %ss, ' % (n, names[int(c)])  # add to string # i.e. 1 crosswalk
				# s += f'{det[:, 4].item():.4f} '
				# print(n)

				# Write results
				for *xyxy, conf, cls in det:
					if save_txt:  # Write to file
						xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
						with open(txt_path + '.txt', 'a') as f:
							x, y, w, h = xywh
							string = f"{int(cls)} {conf.item():.4f} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
							f.write(string)  # label format

					if save_img or view_img:  # Add bbox to image
						label = '%s %.2f' % (names[int(cls)], conf)
						# print(type(im0), im0.shape) array, 720, 1280, 3
						if names[int(cls)] in opt.plot_classes:
							# color = colors[int(cls)]
							color = (255, 85, 33)
							plot_one_box(xyxy, im0, label=label, color=color, line_thickness=5)

			# Print time (inference + NMS)
			prt_str = '%sDone. (%.5fs)' % (s, t2 - t1)
			# print(prt_str)
			os.system(f'echo "{prt_str}" >> {opt.output}/detect.log')

			# Stream results
			if view_img:
				cv2.imshow(p, im0)
				if cv2.waitKey(1) == ord('q'):  # q to quit
					raise StopIteration

			# Save results (image with detections)
			if save_img:
				if dataset.mode == 'images':
					im0 = dp.dmpost(im0, det, det_id=detected_img_id, filename=Path(p).name, names=names)
					cv2.imwrite(save_path, im0)
				else:
					if vid_path != save_path:  # new video
						vid_path = save_path
						if isinstance(vid_writer, cv2.VideoWriter):
							vid_writer.release()  # release previous video writer

						fourcc = 'mp4v'  # output video codec
						fps = vid_cap.get(cv2.CAP_PROP_FPS)
						w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
						h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
						vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
					# print(detected_img_id, p, txt_path)
					tmp_filename = Path(txt_path).stem
					im0 = dp.dmpost(im0, det, det_id=detected_img_id, filename=tmp_filename, names=names)
					vid_writer.write(im0)
			detected_img_id += 1
		bar.set_description(f'inf_time: {infer_time*1000:.2f}ms {prt_str:<40}')

	if save_txt or save_img:
		print('Results saved to %s' % out)
		if platform == 'darwin' and not opt.update:  # MacOS
			os.system('open ' + save_path)

	print('Done. (%.3fs)' % (time.time() - t0))
	time_arr = np.array(time_list)
	prnt = f'Done. Network mean inference time: {np.mean(time_arr)*1000:.2f}ms,  Mean FPS: {1/np.mean(time_arr):.2f}.'
	print(f'\nModel size {opt.img_size} inference {prnt}')
	os.system(f'echo "{prnt}" >> {opt.output}/detect.log')
	os.system(f'echo "useroi {opt.img_size} {prnt}" >> detect2.log')


def run(opt, dp):
	with torch.no_grad():
		if opt.update:  # update all models (to fix SourceChangeWarning)
			for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
				detect()
				strip_optimizer(opt.weights)
		else:
			detect(opt, dp)


def get_opt():
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights', nargs='+', type=str, required=False,
						default='runs/SEm_NST1_fog1_ep100/weights/best.pt',
						help='trained model path model.pt ddpath(s)')
	parser.add_argument('--source', type=str, default='example/images', help='source')  # file/folder, 0 for webcam
	parser.add_argument('--output', type=str, default='example/output', help='output folder')  # output folder
	parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
	parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
	parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
	parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
	parser.add_argument('--view-img', action='store_true', help='display results')
	parser.add_argument('--save-txt', type=bool, default=True, help='save results to *.txt')
	parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
	parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
	parser.add_argument('--augment', action='store_true', help='augmented inference')
	parser.add_argument('--update', action='store_true', help='update all models')
	parser.add_argument('--control-line-setting', type=str, default='settings/cl_setting.yaml',
						help='control line setting')
	parser.add_argument('--select-control-line', type=str, default='general',
						help='select which control line. i.e. general, 0036')
	parser.add_argument('--field-size', type=int, default=5, help='receptive field size for post')
	parser.add_argument('--plot-classes', type=list, default=['crosswalk'],
						help='specifies which classes will be drawn')
	parser.add_argument('--not-use-ROI', action='store_true',
						help='not use roi for accelerate inference speed if there is the flag')
	parser.add_argument('--not-use-SSVM', action='store_true',
						help='not use ssvm method for analyse vehicle crossing behavior if there is the flag')

	opt = parser.parse_args()
	return opt


if __name__ == '__main__':
	opt = get_opt()
	opt.weights = 'runs/m_ep300/weights/best.pt'
	opt.not_use_ROI = True
	opt.not_use_SSVM = True

	opt.use_roi = not opt.not_use_ROI
	opt.use_ssvm = not opt.not_use_SSVM
	for_paper = False
	if for_paper:
		exps = [
			'm_ep300', 'SEm_NST0_fog0_ep100', 'SEm_NST1_fog0_ep100',
			'SEm_NST1_fog0_ep300', 'SEm_NST1_fog1_ep100']
		exp = exps[1]

		roi = int(opt.use_roi)
		ssvm = int(opt.use_ssvm)

		opt.weights = f"runs/{exp}/weights/best.pt"
		opt.source = "/home/zzd/datasets/crosswalk/testsets_1770/Images"
		opt.output = f"/home/zzd/datasets/crosswalk/testsets_1770/{exp}_sz{opt.img_size}_ROI{roi}_SSVM{ssvm}"

	dp = post.DmPost(opt)

	print(opt)
	runtime = time.time()
	run(opt, dp)
	print(f'Total runtime: {time.time()-runtime:.5f}')


