import numpy as np
from PIL import Image
import cv2


def inpainting(img, depth, mask, size=(640, 480), INPAINT=cv2.INPAINT_TELEA):  # cv2.INPAINT_NS

	img = np.array(Image.fromarray(img).resize(size, 0))
	mask = np.array(Image.fromarray(mask).resize(size, 0))
	depth = np.array(Image.fromarray(depth).resize(size, 0))

	mask = np.where(mask >= 0.5, 0, 255).astype(np.uint8)
	depth_max, depth_min = depth[mask == 0].max(), depth[mask == 0].min()
	depth = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)

	img_inpaint = cv2.inpaint(img, mask, 3, INPAINT)
	depth_inpaint = cv2.inpaint(depth, mask, 3, INPAINT)
	depth = (depth_inpaint / 255.) * (depth_max - depth_min) + depth_min

	return img_inpaint, depth
