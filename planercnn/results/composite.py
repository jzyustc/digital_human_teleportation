import numpy as np
import cv2
from pymatting import blend, estimate_foreground_ml

bg_path = "inpaint.png"
fg_path = "1.jpeg"
mask_path = "1_result.png"

fg_size = (288, 288)
bg_size = (640, 480)
center = (156, 240)
z = -2

fg = cv2.resize(cv2.imread(fg_path), fg_size).astype(np.float) / 255.
mask = cv2.resize(cv2.imread(mask_path), fg_size).astype(np.float)[:, :, 0] / 255.

bg = cv2.resize(cv2.imread(bg_path), bg_size).astype(np.float) / 255.

fg = estimate_foreground_ml(fg, mask)
image = (blend(fg, bg, mask) * 255).astype(np.uint8)

cv2.imwrite("image.png", image)
