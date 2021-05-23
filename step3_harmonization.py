import cv2
import os
import numpy as np
from iDIH.harmonization import harmonization

'''
STEP 3 : harmonization of the fg and bg
'''


def step3_harmonization(output_name, human_height, image_height, center_pos, z, f=-1, model_type="hrnet18s_idih256",
						checkpoint="iDIH/results/models/hrnet18s_idih256.pth", fg_name="fg.png", bg_name="bg.png",
						mask_name="mask.png", depth_name="depth.npy", results_path="results/", ):
	print('STEP 3')
	print('Harmonization : ', fg_name, bg_name)

	ratio = human_height / image_height

	# position of the fg
	output_path = os.path.join(results_path, output_name)
	output = harmonization(cv2.imread(os.path.join(results_path, fg_name)),
						   cv2.imread(os.path.join(results_path, bg_name)),
						   cv2.imread(os.path.join(results_path, mask_name)), output_path,
						   center_pos=center_pos, fg_ratio=ratio, depth=np.load(os.path.join(results_path, depth_name)),
						   z=z, f=f, model_type=model_type, checkpoint=checkpoint)

	cv2.imwrite(output_path, output)

	print("Harmonization : saved : ", output_name)

	return output


if __name__ == "__main__":
	# ---- settings ----

	# define path
	results_path = "results/"  # change it
	output_name = "output.png"  # change it

	# define params
	human_height = 1.0  # change it
	image_height = 0.3  # change it
	center_pos = (-80, -80)  # change it
	z = -4  # change it
	f = -1  # change it

	output = step3_harmonization(output_name, human_height, image_height, center_pos, z, f=f, results_path=results_path)
