# data loader
from __future__ import print_function, division
import torch
from skimage import transform
import numpy as np
import random


class RescaleT(object):
	'''
	resize as a Square
	'''

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size

	def __call__(self, sample):
		image, label = sample['image'], sample['label']

		img = transform.resize(image, (self.output_size, self.output_size), mode='constant')
		lbl = transform.resize(label, (self.output_size, self.output_size), mode='constant', order=0,
							   preserve_range=True)

		if "trimap" in sample:
			trimap = transform.resize(sample["trimap"], (self.output_size, self.output_size), mode='constant', order=0,
									  preserve_range=True)
			return {'image': image, 'label': label, 'trimap': trimap}

		return {'image': img, 'label': lbl}


class Rescale(object):
	'''
	resize by original ratio of height and weight
	'''

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size

	def __call__(self, sample):
		image, label = sample['image'], sample['label']

		if random.random() >= 0.5:
			image = image[::-1]
			label = label[::-1]

		h, w = image.shape[:2]

		if h > w:
			new_h, new_w = self.output_size * h / w, self.output_size
		else:
			new_h, new_w = self.output_size, self.output_size * w / h

		new_h, new_w = int(new_h), int(new_w)

		# resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		img = transform.resize(image, (new_h, new_w), mode='constant')
		lbl = transform.resize(label, (new_h, new_w), mode='constant', order=0, preserve_range=True)

		if "trimap" in sample:
			trimap = transform.resize(sample["trimap"], (new_h, new_w), mode='constant', order=0, preserve_range=True)
			return {'image': image, 'label': label, 'trimap': trimap}

		return {'image': img, 'label': lbl}


class RandomCrop(object):
	'''
	Crop the image randomly
	'''

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size

	def __call__(self, sample):
		image, label = sample['image'], sample['label']

		if random.random() >= 0.5:
			image = image[::-1]
			label = label[::-1]

		h, w = image.shape[:2]
		new_h, new_w = self.output_size

		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)

		image = image[top: top + new_h, left: left + new_w]
		label = label[top: top + new_h, left: left + new_w]

		if "trimap" in sample:
			trimap = sample["trimap"][:, top: top + new_h, left: left + new_w]
			return {'image': image, 'label': label, 'trimap': trimap}

		return {'image': image, 'label': label}


class ToTensor(object):
	'''
	Convert ndarrays in sample to Tensors.
	'''

	def __call__(self, sample):

		image, label = sample['image'], sample['label']

		tmpLbl = np.zeros(label.shape)

		if (np.max(label) < 1e-6):
			label = label
		else:
			label = label / np.max(label)

		tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
		image = image / (np.max(image) + 1e-6)
		if image.shape[2] == 1:
			tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
			tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
			tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
		else:
			tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
			tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
			tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

		tmpLbl[:, :, 0] = label[:, :, 0]

		# change the r,g,b to b,r,g from [0,255] to [0,1]
		# transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
		tmpImg = tmpImg.transpose((2, 0, 1))
		tmpLbl = label.transpose((2, 0, 1))

		if "trimap" in sample:
			return {'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl.copy()),
					'trimap': sample["trimap"]}

		return {'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl.copy())}
