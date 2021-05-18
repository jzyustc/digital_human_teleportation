from utils.render import *
from utils.inpainting import *

if __name__ == "__main__":

	# 1. read .ply files
	render = Render(k=580, f=-1, image_range=[[-240, 240], [-320, 320]])
	render.read_ply("results/3d_models/bg.ply")

	# 2. render by points or faces
	render.point_projecting()
	# render.render_by_faces()
	render.render_by_point()

	# 3. inpainting
	render.image = render.image.astype(np.uint8)
	rendered_image, rendered_depth = inpainting(render.image, render.depth, render.mask, (render.w, render.h))
	depth_max, depth_min = rendered_depth.max(), rendered_depth.min()
	depth_image = ((rendered_depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)

	# 4. save output
	Image.fromarray(rendered_image).save("results/bg.png")
	Image.fromarray(depth_image).save("results/bg_depth.png")
	np.save("results/bg.npy", rendered_depth)
