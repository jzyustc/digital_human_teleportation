# data loader
from __future__ import print_function, division
import os
import glob
import torch
from utils.transforms import *
from skimage import io
from scipy import ndimage
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import random
from pymatting import blend, estimate_foreground_ml


def aisegment(dataset_path: str):
	'''
	Get the data path of aisegment dataset (with image_path and label_path)
	'''
	tra_img_name_list = []
	tra_lbl_name_list = []

	# get $DATASET/aisegment/clip_img/* (e.g. 1803290444)
	img_path = os.path.join(dataset_path, 'clip_img' + os.sep)

	for numbers in os.listdir(img_path):
		sub_img_path = os.path.join(img_path, numbers)

		# get $DATASET/aisegment/clip_img/1803290444/* (e.g. clip_00000000)
		for clip_numbers in os.listdir(sub_img_path):

			# get $DATASET/aisegment/clip_img/1803290444/clip_00000000/* (e.g. 1803290444-00000004.jpg)
			real_img_path = os.path.join(sub_img_path, clip_numbers)

			for name in os.listdir(real_img_path):

				# get $DATASET/aisegment/matting/1803290444/matting_00000000/1803290444-00000004.png
				img = os.path.join(real_img_path, name)
				label = img.replace("clip_img", "matting").replace("clip", "matting").replace("jpg", "png")

				if os.path.exists(label):
					tra_img_name_list.append(img)
					tra_lbl_name_list.append(label)

	return tra_img_name_list, tra_lbl_name_list


def paixin_comp_mod(dataset_path: str):
	'''
	Get the data path of paixin_comp_mod dataset (with image_path, label_path)
	and ImageNet validation results' path as backgrounds path (background_path)
	'''
	tra_img_name_list = []
	tra_lbl_name_list = []
	tra_bg_name_list = []
	tra_ez_bg_name_list = []

	# paixin reptile gets 4 kinds of image
	sub_list = ["man_in_white", "man_with_pose", "mixed", "woman_with_food"]

	for sub in sub_list:

		# get $DATASET/paixin_comp_mod/* (e.g. mixed)
		sub_path = os.path.join(dataset_path, sub)

		if not os.path.exists(sub_path): continue

		img_path = os.path.join(sub_path, 'raw_image' + os.sep)
		mask_path = os.path.join(sub_path, 'mask_image' + os.sep)

		for name in os.listdir(img_path):

			# get $DATASET/paixin_comp_mod/raw_image/* (e.g. 1000124_308057842_EDUm9T5o.png)
			img_name = os.path.join(img_path, name)
			mask_name = os.path.join(mask_path, name)

			if os.path.exists(mask_name):
				tra_img_name_list += [img_name]
				tra_lbl_name_list += [mask_name]

	# get $DATASET/background_easy/* (e.g. 1.png)
	background_easy_path = os.path.join(dataset_path, "background_easy" + os.sep)
	# change this when the image format is not png
	tra_ez_bg_name_list += glob.glob(background_easy_path + "/*.png")

	# get $DATASET/background/* (e.g. ILSVRC2012_val_00000001.JPEG)
	background_path = os.path.join(dataset_path, "background" + os.sep)
	# change this when the image format is not JPEG
	tra_bg_name_list += glob.glob(background_path + "/*.JPEG")

	return tra_img_name_list, tra_lbl_name_list, tra_bg_name_list, tra_ez_bg_name_list


class AisegmentDataset(Dataset):
	'''
	Dataset of Aisegment
	'''

	def __init__(self, img_name_list, lbl_name_list, transform=None):
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		self.transform = transform

	def __len__(self):
		return len(self.image_name_list)

	def load(self):
		'''
		To load all of the results before training
		- just feed self.load() to the dataloader
		'''
		data = []
		for idx in range(len(self.image_name_list)):
			image_path = self.image_name_list[idx]
			label_path = self.label_name_list[idx]

			try:

				image = io.imread(image_path) / 255.
				label = io.imread(label_path) / 255.

				# label : rgb 2 gray
				if len(label.shape) == 3:
					label = label[:, :, -1]

				# label : transform to (h,w,1)
				label = label[:, :, np.newaxis]
				if 2 == len(image.shape): image = image[:, :, np.newaxis]

				sample = {'image': image, 'label': label}

				if self.transform:
					sample = self.transform(sample)

				data.append(sample)

			except:
				print("ERROR in ", image_path)

		return data

	def __getitem__(self, idx):
		'''
		To load all of the results before training
		- just feed self.load() to the dataloader
		'''

		while True:
			# because of some error occurs when uploading my dataset, we use this
			# to prevent of loading error
			try:
				image = io.imread(self.image_name_list[idx]) / 255.
				label = io.imread(self.label_name_list[idx]) / 255.
				break
			except:
				idx += 1

		# label : rgb 2 gray
		if len(label.shape) == 3:
			label = label[:, :, -1]

		# label : transform to (h,w,1)
		label = label[:, :, np.newaxis]
		if 2 == len(image.shape): image = image[:, :, np.newaxis]

		sample = {'image': image, 'label': label}

		if self.transform:
			sample = self.transform(sample)

		return sample


class PaixinDataset(Dataset):
	'''
	Dataset of paixin_comp_mod
	warning : since we use the foreground estimation function for image, it would be much slower without load_once
	'''

	def __init__(self, img_name_list, lbl_name_list, bg_name_list, ez_bg_name_list, transform=None, load_once=False,
				 update_ratio_epoch=50):
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		self.bg_name_list = bg_name_list
		self.ez_bg_name_list = ez_bg_name_list
		self.transform = transform

		self.sample_list = []
		self.bg_list = []
		self.ez_bg_list = []

		self.load_once = load_once

		# load the foreground information and background results before training
		if load_once: self.load()

		# use this to mark the training epoch, for mixing of bg and ez_bg
		self.epoch = 0
		self.update_ratio_epoch = update_ratio_epoch

	def __len__(self):
		return len(self.image_name_list)

	def generate_trimap(self, alpha, value=5, low=None, high=None):
		'''
		generate trimap by alpha image, using distance strategy
		'''

		# low and high are used in distance strategy
		if low is None: low = 1
		if high is None: high = max(max(alpha.shape[0], alpha.shape[1]) // 160, 10)

		# formulate the alpha
		alpha = np.where(alpha > 255 - value, 255, alpha)
		alpha = np.where(alpha < value, 0, alpha)

		# get foreground and background part by alpha
		fg = np.array(np.equal(alpha, 255).astype(np.float32))
		unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))  # unknown = alpha > 0
		unknown = unknown - fg

		# compute unknown part by distance
		unknown = ndimage.morphology.distance_transform_edt(unknown == 0) <= np.random.randint(low, high)

		trimap = torch.zeros(alpha.shape[0], alpha.shape[1])
		trimap[unknown] = 1.
		return trimap.unsqueeze(0)

	def load_one_foreground(self, idx):
		image = io.imread(self.image_name_list[idx]) / 255.
		label = io.imread(self.label_name_list[idx]) / 255.

		# scale results
		sample0 = Rescale(320)({'image': image, 'label': label})
		image, label = sample0["image"], sample0["label"]

		# estimate foreground by multi-level method
		foreground = estimate_foreground_ml(image, label)

		if len(label.shape) == 3:
			label = label[:, :, -1]

		# get trimap by alpha
		trimap = self.generate_trimap(np.array(label * 255, dtype=np.uint8))

		# label : transform to (h,w,1)
		label = label[:, :, np.newaxis]
		if 2 == len(foreground.shape): foreground = foreground[:, :, np.newaxis]

		sample = {'image': foreground, 'label': label, 'trimap': trimap}

		return sample

	def load(self):
		'''
		load the foreground information and background results before training
		'''

		# load foreground information
		for idx in range(len(self.image_name_list)):
			image_path = self.image_name_list[idx]
			label_path = self.label_name_list[idx]
			try:
				if not os.path.exists(label_path):
					print("ERROR : not find Matting Match for ", image_path)
					continue
				sample = self.load_one_foreground(idx)
				self.sample_list.append(sample)
			except:
				print("ERROR in ", image_path)

		# load background results
		for idx in range(len(self.bg_name_list)):
			bg_path = self.bg_name_list[idx]
			try:
				bg = io.imread(bg_path)
				self.bg_list.append(Image.fromarray(bg))
			except:
				print("ERROR in ", bg_path)

		# load easy background results
		for idx in range(len(self.ez_bg_name_list)):
			ez_bg_path = self.ez_bg_name_list[idx]
			try:
				ez_bg = io.imread(ez_bg_path)
				self.ez_bg_list.append(Image.fromarray(ez_bg))
			except:
				print("ERROR in ", ez_bg_path)

	def mixing_ratio(self):
		# add 0.1 to ratio every update_ratio_epoch epochs
		return min((self.epoch // self.update_ratio_epoch) * 0.1, 1.)

	def __getitem__(self, idx):
		'''
		compositing the foreground and background information when training
		'''

		if self.load_once:
			sample = self.sample_list[idx]
			bg = self.bg_list[random.randint(0, len(self.bg_list) - 1)]
			ez_bg = self.ez_bg_list[random.randint(0, len(self.ez_bg_list) - 1)]
		else:
			sample = self.load_one_foreground(idx)
			bg_idx = random.randint(0, len(self.bg_name_list) - 1)
			bg = Image.fromarray(io.imread(self.bg_name_list[bg_idx]))
			ez_bg_idx = random.randint(0, len(self.ez_bg_name_list) - 1)
			ez_bg = Image.fromarray(io.imread(self.ez_bg_name_list[ez_bg_idx]))

		foreground, label = sample["image"], sample["label"]

		# choose background image randomly, then transform it
		bg = bg.resize((foreground.shape[1], foreground.shape[0]), Image.ANTIALIAS)
		bg = np.array(bg, dtype=np.float32) / 255.
		if 2 == len(bg.shape): bg = bg[:, :, np.newaxis]

		# choose easy background image randomly, then transform it
		ez_bg = ez_bg.resize((foreground.shape[1], foreground.shape[0]), Image.ANTIALIAS)
		ez_bg = np.array(ez_bg, dtype=np.float32) / 255.
		if 2 == len(ez_bg.shape): ez_bg = ez_bg[:, :, np.newaxis]

		# train with bg from easy to hard
		mixed_ratio = self.mixing_ratio()
		mixed_bg = mixed_ratio * bg + (1 - mixed_ratio) * ez_bg

		# blend the foreground and background by alpha compositing
		new_image = blend(foreground, mixed_bg, label)
		sample["image"] = new_image

		if self.transform:
			sample = self.transform(sample)

		return sample
