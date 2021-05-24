import os
from planercnn.utils_3d.render import *
from planercnn.utils_3d.inpainting import *

'''
STEP 2 : render the 3d model to a single rgb background
'''


def step2_ply2bg(ply_sub_path, k=580, f=-1, image_range=None, method="quick", bg_name="bg.png", depth_name="depth.npy",
				 bg_depth_name="bg_depth.png", results_path="results/", **kwargs):
	print('STEP 2')
	print('Render : ', ply_sub_path)

	if image_range is None:
		image_range = [[-240, 240], [-320, 320]]

	# 1. read .ply files
	render = Render(k=k, f=f, image_range=image_range)
	render.read_ply(os.path.join(results_path, ply_sub_path))

	# 2. render by points or faces
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

	# 3. inpainting
	render.image = render.image.astype(np.uint8)
	rendered_image, rendered_depth = inpainting(render.image, render.depth, render.mask, (render.w, render.h))
	depth_max, depth_min = rendered_depth.max(), rendered_depth.min()
	depth_image = ((rendered_depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)

	# 4. save output
	Image.fromarray(rendered_image).save(os.path.join(results_path, bg_name))
	print("Render : background saved : ", os.path.join(results_path, bg_name))

	np.save(os.path.join(results_path, depth_name), rendered_depth)
	print("Render : depth saved : ", os.path.join(results_path, depth_name))

	if bg_depth_name != "":
		Image.fromarray(depth_image).save(os.path.join(results_path, bg_depth_name))
		print("Render : depthmap saved : ", os.path.join(results_path, bg_depth_name))

	'''
	'''
	Image.fromarray((render.mask * 255).astype(np.uint8)).save(os.path.join(results_path, "step2_inter_mask.png"))
	print("Render : mask saved : ", os.path.join(results_path, "step2_inter_mask.png"))

	Image.fromarray(render.image).save(os.path.join(results_path, "step2_inter_no_inpainting.png"))
	print("Render : not inpainting saved : ", os.path.join(results_path, "step2_inter_no_inpainting.png"))

	depth_max, depth_min = render.depth.max(), render.depth.min()
	depth_image = ((render.depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
	Image.fromarray(depth_image).save(os.path.join(results_path, "step2_inter_no_depth.png"))
	print("Render : no inpainting depth saved : ", os.path.join(results_path, "step2_inter_no_depth.png"))


if __name__ == "__main__":
	# ---- settings ----

	# define path
	results_path = "results/"  # change it
	ply_sub_path = "3d_models/bg_after.ply"  # change it

	# define params
	k = 580  # change it
	f = -1  # change it
	image_range = [[-240, 240], [-320, 320]]  # change it
	method = "quick"  # change it

	step2_ply2bg(ply_sub_path, k=k, f=f, image_range=image_range, method=method, results_path=results_path)
