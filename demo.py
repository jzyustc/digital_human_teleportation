import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pymatting import *
import streamlit as st

from MODNet.models.models.modnet import MODNet
from planercnn.utils_3d.render import *
from planercnn.utils_3d.inpainting import *
from iharm.inference.predictor import Predictor
from iharm.inference.utils import load_model, find_checkpoint
from iharm.mconfigs import ALL_MCONFIGS
from iharm.utils.exp import load_config_file

use_gpu = torch.cuda.is_available()
device = torch.device("cuda") if use_gpu else torch.device("cpu")


def load_modnet():
	print("load modnet")
	use_gpu = torch.cuda.is_available()
	device = torch.device("cuda") if use_gpu else torch.device("cpu")

	# step1 :
	modnet = MODNet(backbone_pretrained=False).to(device)
	modnet = nn.DataParallel(modnet)
	if torch.cuda.is_available():
		modnet.load_state_dict(torch.load("MODNet/results/models/modnet_photographic_portrait_matting.ckpt"))
	else:
		modnet.load_state_dict(torch.load("MODNet/results/models/modnet_photographic_portrait_matting.ckpt",
										  map_location=torch.device('cpu')))
	modnet.eval()

	return modnet


def render_read_ply(ply):
	print("render read ply")
	render = Render(k=580, f=-1, image_range=[[-240, 240], [-320, 320]])
	render.read_ply(os.path.join(results_path, ply))
	return render


def load_idih():
	print("load iDIH")

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

	args, cfg = parse_args()
	args.model_type = "hrnet18s_idih256"
	args.checkpoint = "iDIH/results/models/hrnet18s_idih256.pth"

	checkpoint_path = find_checkpoint(cfg.MODELS_PATH, args.checkpoint)
	net = load_model(args.model_type, checkpoint_path)

	predictor = Predictor(net, device)

	return predictor


def step1_matting(fg_image, modnet, ref_size=512):
	print("step1")
	# load image
	im = np.asarray(fg_image)

	# reformat to rgb-channels
	if len(im.shape) == 2:
		im = im[:, :, None]
	if im.shape[2] == 1:
		im = np.repeat(im, 3, axis=2)
	elif im.shape[2] == 4:
		im = im[:, :, :3]

	# define image to tensor transform
	im_transform = transforms.Compose(
		[
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		]
	)
	# convert image to PyTorch tensor
	im = im_transform(Image.fromarray(im))

	# add mini-batch dim
	im = im[None, :, :, :]

	# resize image for input
	im_b, im_c, im_h, im_w = im.shape
	if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
		if im_w >= im_h:
			im_rh = ref_size
			im_rw = int(im_w / im_h * ref_size)
		else:
			im_rw = ref_size
			im_rh = int(im_h / im_w * ref_size)
	else:
		im_rh = im_h
		im_rw = im_w

	im_rw = im_rw - im_rw % 32
	im_rh = im_rh - im_rh % 32
	im_raw = im.clone()
	im = nn.Upsample(size=(im_rh, im_rw), mode='area')(im)

	# inference
	_, _, mask = modnet(im.to(device), True)

	# ------- 3. output --------

	# resize and save mask
	mask = nn.Upsample(size=(im_h, im_w), mode='area')(mask)
	mask = mask[0, 0].data.cpu().numpy()

	# save the foreground after multi-level foreground estimation
	im = im_raw[0].permute(1, 2, 0).data.cpu().numpy() * 0.5 + 0.5
	foreground = estimate_foreground_ml(im, mask)

	# mask, fg
	return Image.fromarray(((mask * 255).astype('uint8')), mode='L'), \
		   Image.fromarray((foreground * 255).astype(np.uint8))


def step2_ply2bg(render, method="quick", **kwargs):
	print("step2")
	if method == "quick":
		if "size" not in kwargs:
			size = 3
		else:
			size = kwargs["size"]
		render.quick_render(size)  # quick render
	else:
		render.point_projecting()  # project is needed if use render_by_faces/point

		if method == "face":
			render.render_by_faces()
		elif method == "point":
			render.render_by_point()
		else:
			print("Render : Illegal method : ", method)
			exit()

	render.image = render.image.astype(np.uint8)
	rendered_image, rendered_depth = inpainting(render.image, render.depth, render.mask, (render.w, render.h))
	depth_max, depth_min = rendered_depth.max(), rendered_depth.min()
	depth_image = ((rendered_depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)

	# bg ,depth ,bg_depth
	return Image.fromarray(rendered_image), rendered_depth, Image.fromarray(depth_image)


def step3_harmonization(predictor, fg, bg, mask, human_height, image_height, center_pos, z, f=-1):
	print("step3")
	fg_ratio = human_height / image_height

	# from uint8 [0,255] to float32 [0,1.]
	fg = np.array(fg).astype(np.float) / 255.
	mask = np.array(mask).astype(np.float) / 255.
	bg = np.array(bg).astype(np.float) / 255.
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
		return
	elif max(r - l, b - t) >= min(bg_shape[0], bg_shape[1]):
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

	mask = mask[:, :, np.newaxis].repeat(3, axis=2)
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

	return Image.fromarray(output)


# cache
modnet_func = st.cache(load_modnet, allow_output_mutation=True)
render_func = st.cache(render_read_ply, allow_output_mutation=True)
idih_func = st.cache(load_idih, allow_output_mutation=True)
step1_func = st.cache(step1_matting, allow_output_mutation=True)
step2_func = st.cache(step2_ply2bg, allow_output_mutation=True)
step3_func = st.cache(step3_harmonization, allow_output_mutation=True)

# changeable params

z = st.sidebar.slider('z : depth of the human (m)', min_value=-5., max_value=-1., value=-1.8, step=0.01)

human_height = st.sidebar.slider('human height (m)', min_value=0., max_value=2., value=1., step=0.01)
image_height = st.sidebar.slider('image height (m)', min_value=0., max_value=3., value=1.4, step=0.01)

image_range = [[-240, 240], [-320, 320]]

center_pos_x = st.sidebar.slider('human pos x', min_value=image_range[0][0], max_value=image_range[0][1], value=-70,
								 step=1)
center_pos_y = st.sidebar.slider('human pos y', min_value=image_range[1][0], max_value=image_range[1][1], value=90,
								 step=1)
center_pos = (center_pos_x, center_pos_y)  # position of the human (pixels)

k = st.sidebar.slider('k : the size of pixels', min_value=0, max_value=1000, value=580, step=1)

# global
results_path = "results/"
modnet = modnet_func()
idih = idih_func()

# inputs
ply_sub_path = os.path.join("3d_models/", st.selectbox(".ply filename", os.listdir("results/3d_models/")))
render = render_func(ply_sub_path)
render.k = k

fg_image = st.file_uploader("Upload an image", type="jpg")

# run
if fg_image is None:
	default_human = os.path.join(results_path, "trump.jpg")
	fg_image = Image.open(default_human)
else:
	fg_image = Image.open(fg_image)

fg_size = fg_image.size
st.image(fg_image, caption='Input image.', use_column_width=True)

mask, fg = step1_func(fg_image, modnet)
bg, depth, bg_depth = step2_func(render)

st.image(bg, caption='Background', use_column_width=True)

output = step3_func(idih, fg, bg, mask, human_height, image_height, center_pos, z)
if output is not None:
	st.image(output, caption='Output', use_column_width=True)
else:
	st.write("Illegal params. Try to make the human height smaller")
