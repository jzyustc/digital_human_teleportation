import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F

import glob

from models.model import *
from models.refinement_net import RefineModel
from datasets.inference_dataset import InferenceDataset
from evaluate_utils import *
from plane_utils import *
from options import parse_args
from config import InferenceConfig


class PlaneRCNNDetector():
	def __init__(self, options, config, modelType, checkpoint_dir=''):
		self.options = options
		self.config = config
		self.modelType = modelType
		self.model = MaskRCNN(config)
		self.model.cuda()
		self.model.eval()

		checkpoint_dir = 'checkpoint/planercnn_' + options.anchorType + '_' + options.suffix

		## Indicates that the refinement network is trained separately

		self.model.load_state_dict(torch.load(checkpoint_dir + '/checkpoint.pth'))

		self.refine_model = RefineModel(options)

		self.refine_model.cuda()
		self.refine_model.eval()

		state_dict = torch.load(checkpoint_dir + '/checkpoint_refine.pth')
		self.refine_model.load_state_dict(state_dict)

		return

	def detect(self, sample):
		camera = sample[30][0].cuda()

		images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, gt_parameters, gt_depth, extrinsics, planes, gt_segmentation = \
			sample[0].cuda(), sample[1].numpy(), sample[2].cuda(), sample[3].cuda(), sample[4].cuda(), sample[5].cuda(), \
			sample[6].cuda(), sample[7].cuda(), sample[8].cuda(), sample[9].cuda(), sample[10].cuda(), sample[11].cuda()

		rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, target_parameters, mrcnn_parameters, detections, detection_masks, detection_gt_parameters, detection_gt_masks, rpn_rois, roi_features, roi_indices, depth_np_pred = self.model.predict(
			[images, image_metas, gt_class_ids, gt_boxes, gt_masks, gt_parameters, camera],
			mode='inference_detection', use_nms=2, use_refinement=True)

		if len(detections) > 0:
			detections, detection_masks = unmoldDetections(self.config, camera, detections, detection_masks,
														   depth_np_pred, debug=False)
			pass

		XYZ_pred, detection_mask, plane_XYZ = calcXYZModule(self.config, camera, detections, detection_masks,
															depth_np_pred, return_individual=True)
		detection_mask = detection_mask.unsqueeze(0)

		input_dict = {'image': images, 'depth': gt_depth, 'mask': gt_masks, 'bbox': gt_boxes, 'extrinsics': extrinsics,
					  'segmentation': gt_segmentation, 'camera': camera}

		detection_dict = {'XYZ': XYZ_pred, 'depth': XYZ_pred[1:2], 'mask': detection_mask, 'detection': detections,
						  'masks': detection_masks, 'depth_np': depth_np_pred, 'plane_XYZ': plane_XYZ}

		camera = camera.unsqueeze(0)

		new_input_dict = {k: v for k, v in input_dict.items()}
		new_input_dict['image'] = (input_dict['image'] + self.config.MEAN_PIXEL_TENSOR.view(
			(-1, 1, 1))) / 255.0 - 0.5
		new_input_dict['image_2'] = (sample[13].cuda() + self.config.MEAN_PIXEL_TENSOR.view(
			(-1, 1, 1))) / 255.0 - 0.5
		detection_masks = detection_dict['masks']
		depth_np = detection_dict['depth_np']
		image = new_input_dict['image']
		image_2 = new_input_dict['image_2']

		masks_inp = torch.cat([detection_masks.unsqueeze(1), detection_dict['plane_XYZ']], dim=1)

		image = torch.nn.functional.interpolate(image[:, :, 80:560], size=(192, 256), mode='bilinear')
		image_2 = torch.nn.functional.interpolate(image_2[:, :, 80:560], size=(192, 256), mode='bilinear')
		masks_inp = torch.nn.functional.interpolate(masks_inp[:, :, 80:560], size=(192, 256), mode='bilinear')
		depth_np = torch.nn.functional.interpolate(depth_np[:, 80:560].unsqueeze(1), size=(192, 256),
												   mode='bilinear').squeeze(1)
		plane_depth = torch.nn.functional.interpolate(detection_dict['depth'][:, 80:560].unsqueeze(1),
													  size=(192, 256), mode='bilinear').squeeze(1)

		new_input_dict['image'] = image
		new_input_dict['image_2'] = image_2

		results = self.refine_model(image, image_2, camera, masks_inp, detection_dict['detection'][:, 6:9],
									plane_depth, depth_np)

		masks = results[-1]['mask'].squeeze(1)

		all_masks = torch.softmax(masks, dim=0)

		all_masks = torch.nn.functional.interpolate(all_masks.unsqueeze(1), size=(480, 640),
													mode='bilinear').squeeze(1)
		all_masks = (all_masks.max(0, keepdim=True)[1] == torch.arange(len(all_masks)).cuda().long().view(
			(-1, 1, 1))).float()
		masks = all_masks[1:]
		detection_masks = torch.zeros(detection_dict['masks'].shape).cuda()
		detection_masks[:, 80:560] = masks

		detection_dict['masks'] = detection_masks
		detection_dict['mask'][:, 80:560] = (
				masks.max(0, keepdim=True)[0] > (1 - masks.sum(0, keepdim=True))).float()

		return [detection_dict]


def evaluate(options):
	config = InferenceConfig(options)
	config.FITTING_TYPE = options.numAnchorPlanes

	image_list = glob.glob(options.customDataFolder + '/*.png') + glob.glob(options.customDataFolder + '/*.jpg')
	camera = np.zeros(6)
	with open(options.customDataFolder + '/camera.txt', 'r') as f:
		for line in f:
			values = [float(token.strip()) for token in line.split(' ') if token.strip() != '']
			for c in range(6):
				camera[c] = values[c]
				continue
			break
	dataset = InferenceDataset(options, config, image_list=image_list, camera=camera)
	dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

	with torch.no_grad():
		detector = PlaneRCNNDetector(options, config, modelType='final')

	os.system('rm ' + options.test_dir + '/*_final.png')

	for sampleIndex, sample in enumerate(dataloader):

		camera = sample[30][0].cuda()

		images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, gt_parameters, gt_depth, extrinsics, planes, gt_segmentation = \
			sample[0].cuda(), sample[1].numpy(), sample[2].cuda(), sample[3].cuda(), sample[4].cuda(), sample[5].cuda(), \
			sample[6].cuda(), sample[7].cuda(), sample[8].cuda(), sample[9].cuda(), sample[10].cuda(), sample[11].cuda()

		masks = (gt_segmentation == torch.arange(gt_segmentation.max() + 1).cuda().view(-1, 1, 1)).float()

		input_pair = [{'image': images, 'depth': gt_depth, 'bbox': gt_boxes, 'extrinsics': extrinsics,
					   'segmentation': gt_segmentation, 'camera': camera, 'plane': planes[0],
					   'masks': masks, 'mask': gt_masks}]

		with torch.no_grad():
			detection_pair = detector.detect(sample)

		for c in range(len(detection_pair)):
			np.save(options.test_dir + '/' + str(sampleIndex % 500) + '_plane_parameters_' + str(c) + '.npy',
					detection_pair[c]['detection'][:, 6:9])
			np.save(options.test_dir + '/' + str(sampleIndex % 500) + '_plane_masks_' + str(c) + '.npy',
					detection_pair[c]['masks'][:, 80:560])
			continue

		for pair_index, (input_dict, detection_dict) in enumerate(zip(input_pair, detection_pair)):
			depth_pred = detection_dict['depth'][0].detach().cpu().numpy()
			cv2.imwrite(options.test_dir + '/{}_depth_0_final.png'.format(sampleIndex % 500),
						drawDepthImage(depth_pred[80:560]))

			depth = depth_pred[80:560]
			print("DEPTH : ", depth.shape, depth.max(), depth.min())

			images = input_dict['image'].detach().cpu().numpy().transpose((0, 2, 3, 1))
			images = unmold_image(images, config)
			image = images[0][80:560]
			print("IMAGE : ", image.shape, image.max(), image.min())

			XYZ = detection_dict["XYZ"].permute(1, 2, 0)[80:560].detach().cpu().numpy()
			print("XYZ", XYZ.shape, XYZ.max(), XYZ.min())

			create_output(image, XYZ, options.test_dir + '/{}_points_cloud.ply'.format(sampleIndex % 500))


# Function to create point cloud file
def create_output(image, XYZ, filename):
	XYZ = XYZ.reshape((-1, 3))
	Z = -XYZ[:, 1].copy()
	XYZ[:, 1] = XYZ[:, 2]
	XYZ[:, 2] = Z
	image = np.insert(image.reshape(-1, 3), 3, values=255, axis=1)
	B = image[:, 0].copy()
	image[:, 0] = image[:, 2]
	image[:, 2] = B
	vertices = np.hstack([XYZ.reshape(-1, 3), image])
	np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d %d')
	ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
'''
	with open(filename, 'r+') as f:
		old = f.read()
		f.seek(0)
		f.write(ply_header % dict(vert_num=len(vertices)))
		f.write(old)
