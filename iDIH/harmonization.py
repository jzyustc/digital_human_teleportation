import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, 'scripts')
from .iharm.inference.predictor import Predictor
from .iharm.inference.utils import load_model, find_checkpoint
from .iharm.mconfigs import ALL_MCONFIGS
from .iharm.utils.exp import load_config_file

from pymatting import blend, estimate_foreground_ml


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_type', choices=ALL_MCONFIGS.keys())
	parser.add_argument('--checkpoint', type=str,
						help='The path to the checkpoint. '
							 'This can be a relative path (relative to cfg.MODELS_PATH) '
							 'or an absolute path. The file extension can be omitted.')
	parser.add_argument(
		'--images', type=str,
		help='Path to directory with .jpg images to get predictions for.'
	)
	parser.add_argument(
		'--masks', type=str,
		help='Path to directory with .png binary masks for images, named exactly like images without last _postfix.'
	)
	parser.add_argument(
		'--resize', type=int, default=256,
		help='Resize image to a given size before feeding it into the network. If -1 the network input is not resized.'
	)
	parser.add_argument(
		'--original-size', action='store_true', default=False,
		help='Resize predicted image back to the original size.'
	)
	parser.add_argument('--gpu', type=str, default=0, help='ID of used GPU.')
	parser.add_argument('--config-path', type=str, default='iDIH/config.yml', help='The path to the config file.')
	parser.add_argument(
		'--results-path', type=str, default='',
		help='The path to the harmonized images. Default path: cfg.EXPS_PATH/predictions.'
	)
	try:
		args = parser.parse_args()
	except:  # Jupyter
		args = parser.parse_known_args()[0]
	cfg = load_config_file(args.config_path, return_edict=True)
	cfg.EXPS_PATH = Path(cfg.EXPS_PATH)
	cfg.RESULTS_PATH = Path(args.results_path) if len(args.results_path) else cfg.EXPS_PATH / 'predictions'
	cfg.RESULTS_PATH.mkdir(parents=True, exist_ok=True)
	return args, cfg


def harmonization(fg, bg, mask, output_path, center_pos, fg_ratio, depth, z, f=-1, model_type="hrnet18s_idih256",
				  checkpoint="iDIH/results/models/hrnet18s_idih256.pth"):
	# ------- 0. get settings --------

	args, cfg = parse_args()
	args.model_type = model_type
	args.checkpoint = checkpoint

	# ------- 1. load model --------

	use_gpu = torch.cuda.is_available()
	device = torch.device(f'cuda:{args.gpu}') if use_gpu else torch.device("cpu")

	checkpoint_path = find_checkpoint(cfg.MODELS_PATH, args.checkpoint)
	net = load_model(args.model_type, checkpoint_path, verbose=True)
	predictor = Predictor(net, device)

	# ------- 2. applying process --------

	# from uint8 [0,255] to float32 [0,1.]
	fg = fg.astype(np.float) / 255.
	mask = mask.astype(np.float) / 255.
	bg = bg.astype(np.float) / 255.

	output = harmonization_by_params(predictor, fg.copy(), bg.copy(), mask.copy(), output_path, center_pos, fg_ratio,
									 depth, z, f=f)

	return output


def harmonization_by_params(predictor, fg, bg, mask, output_path, center_pos, fg_ratio, depth, z, f=-1):
	# reshape to camera plane
	human_height = int(bg.shape[0] * fg_ratio * f / z)
	human_width = int(fg.shape[1] / fg.shape[0] * human_height)
	fg_shape = (human_width, human_height)
	bg_shape = (bg.shape[1], bg.shape[0])

	fg = cv2.resize(fg, fg_shape, cv2.INTER_LINEAR)
	mask = cv2.resize(mask, fg_shape, cv2.INTER_LINEAR)

	# crop the part inside the bg
	l = bg_shape[0] // 2 + center_pos[0] - fg_shape[0] // 2
	r = l + fg_shape[0]
	t = bg_shape[1] // 2 - center_pos[1] - fg_shape[1] // 2
	b = t + fg_shape[1]

	l_crop = max(-l, 0)
	r_crop = max(r - (bg_shape[0] - 1), 0)
	t_crop = max(-t, 0)
	b_crop = max(b - (bg_shape[1] - 1), 0)

	l += l_crop
	r -= r_crop
	t += t_crop
	b -= b_crop

	if l >= r or t >= b:  # no remain image
		cv2.imwrite(output_path, bg * 255.)
		return
	elif max(r - l, b - t) >= min(bg_shape[0], bg_shape[1]):
		print("ERROR : size of foreground image is too big !")
		cv2.imwrite(output_path, bg * 255.)
		return

	fg = fg[t_crop:fg_shape[1] - b_crop, l_crop:fg_shape[0] - r_crop]
	mask = mask[t_crop:fg_shape[1] - b_crop, l_crop:fg_shape[0] - r_crop]

	# square zero padding
	fg_shape_diff = fg.shape[0] - fg.shape[1]
	if fg_shape_diff > 0:  # h > w
		l_remain = min(l, fg_shape_diff)
		padding = ((0, 0), (l_remain, fg_shape_diff - l_remain), (0, 0))
		l -= l_remain
	else:
		t_remain = min(t, -fg_shape_diff)
		padding = ((t_remain, -fg_shape_diff - t_remain), (0, 0), (0, 0))
		t -= t_remain

	fg = np.pad(fg, padding)
	mask = np.pad(mask, padding)

	# pos and shape
	human_size = max(fg.shape[0], fg.shape[1])
	fg_shape = (human_size, human_size)
	r = l + fg_shape[0]
	b = t + fg_shape[1]

	depth_region = cv2.resize(depth[t:b, l:r], (256, 256), cv2.INTER_LINEAR)
	bg_region = cv2.resize(bg[t:b, l:r], (256, 256), cv2.INTER_LINEAR)
	fg = cv2.resize(fg, (256, 256), cv2.INTER_LINEAR)

	# compositing
	mask = cv2.resize(mask[:, :, 0], (256, 256), cv2.INTER_LINEAR)
	mask = mask.astype(np.float32)[:, :, np.newaxis]
	mask[z < depth_region] = 0

	image_composited = blend(fg, bg_region, mask)

	# harmonization
	image_composited = (image_composited * 255).astype(np.uint8)

	pred = predictor.predict(image_composited, mask).astype(np.float)

	# ------- 3. output --------

	# to original background
	output = (bg[:, :, :] * 255).astype(np.uint8)
	output[t:b, l:r] = cv2.resize(pred, fg_shape, cv2.INTER_LINEAR)

	return output
