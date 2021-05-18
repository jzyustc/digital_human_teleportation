import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, 'scripts')
from iharm.inference.predictor import Predictor
from iharm.inference.utils import load_model, find_checkpoint
from iharm.mconfigs import ALL_MCONFIGS
from iharm.utils.exp import load_config_file

from pymatting import blend, estimate_foreground_ml


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('model_type', choices=ALL_MCONFIGS.keys())
	parser.add_argument('checkpoint', type=str,
						help='The path to the checkpoint. '
							 'This can be a relative path (relative to cfg.MODELS_PATH) '
							 'or an absolute path. The file extension can be omitted.')
	parser.add_argument(
		'--results', type=str,
		help='Path to directory with .jpg results to get predictions for.'
	)
	parser.add_argument(
		'--masks', type=str,
		help='Path to directory with .png binary masks for results, named exactly like results without last _postfix.'
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
	parser.add_argument('--config-path', type=str, default='./config.yml', help='The path to the config file.')
	parser.add_argument(
		'--results-path', type=str, default='',
		help='The path to the harmonized results. Default path: cfg.EXPS_PATH/predictions.'
	)

	args = parser.parse_args()
	cfg = load_config_file(args.config_path, return_edict=True)
	cfg.EXPS_PATH = Path(cfg.EXPS_PATH)
	cfg.RESULTS_PATH = Path(args.results_path) if len(args.results_path) else cfg.EXPS_PATH / 'predictions'
	cfg.RESULTS_PATH.mkdir(parents=True, exist_ok=True)
	return args, cfg


# ------- 0. get settings --------

args, cfg = parse_args()

fg_path = "results/fg.png"  # change it
bg_path = "results/bg.png"  # change it
mask_path = "results/mask.png"  # change it

bg_shape = (524, 288)  # change it
fg_shape = (256, 256)  # change it

# ------- 1. load model --------

device = torch.device(f'cuda:{args.gpu}')
checkpoint_path = find_checkpoint(cfg.MODELS_PATH, args.checkpoint)
net = load_model(args.model_type, checkpoint_path, verbose=True)
predictor = Predictor(net, device)

# bottom center
l = bg_shape[0] // 2 - fg_shape[0] // 2
r = bg_shape[0] - l
t = bg_shape[1] - fg_shape[1]
b = bg_shape[1]

# ------- 2. applying process --------

# load
fg = cv2.resize(cv2.imread(fg_path), fg_shape).astype(np.float) / 255.
mask = cv2.resize(cv2.imread(mask_path), fg_shape).astype(np.float)[:, :, 0] / 255.
bg = cv2.resize(cv2.imread(bg_path), bg_shape).astype(np.float) / 255.

bg_region = bg[t:b, l:r]

# compositing
image_composited = blend(fg, bg_region, mask)

# harmonization
image_composited = (image_composited * 255).astype(np.uint8)

mask_image = cv2.imread(mask_path)
mask_image = cv2.resize(mask_image, fg_shape, cv2.INTER_LINEAR)
mask = mask_image[:, :, 0]
mask[mask <= 100] = 0
mask[mask > 100] = 1
mask = mask.astype(np.float32)[:, :, np.newaxis]

pred = predictor.predict(image_composited, mask).astype(np.float)

# ------- 3. output --------

# to original background
output = (bg[:, :, :] * 255).astype(np.uint8)
output[t:b, l:r] = pred

cv2.imwrite("results/output_composited.png", image_composited)
print("saved : results/output_composited.png")

cv2.imwrite("results/output_harmonized.png", pred)
print("saved : results/output_harmonized.png")

cv2.imwrite("results/output.png", output)
print("saved : results/output.png")
