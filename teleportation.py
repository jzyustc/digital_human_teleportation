from step1_matting import step1_matting
from step2_ply2bg import step2_ply2bg
from step3_harmonization import step3_harmonization

if __name__ == "__main__":
	# define path
	results_path = "results/"  # result path
	fg_image_name = "human.jpg"  # foreground image
	output_name = "output.png"  # output path
	ply_sub_path = "3d_models/bg_after.ply"  # .ply file generated from  ./pre_bg2ply.py

	# step 2 params
	k = 580  # the sise of pixel (e.g. 580 pixels/m)
	f = -1  # depth of the camera plane
	image_range = [[-240, 240], [-320, 320]]  # change it
	method = "quick"  # method to render

	# step 3
	human_height = 1.8  # height of human image
	center_pos = (160, 30)  # position to teleport human
	z = -6  # depth to teleport human

	step1_matting(fg_image_name, results_path=results_path)
	step2_ply2bg(ply_sub_path, k=k, f=f, image_range=image_range, method=method, results_path=results_path)
	step3_harmonization(output_name, human_height, (image_range[0][1] - image_range[0][0]) / k, center_pos, z, f=f,
						results_path=results_path)

	print("Teleportation End")
