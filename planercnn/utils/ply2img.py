import numpy as np
import open3d as o3d
from PIL import Image


def show_ply(ply_path, front=None, lookat=None, up=None, zoom=None):
	pcd = o3d.io.read_point_cloud(ply_path)

	vis = o3d.visualization.VisualizerWithEditing()
	vis.create_window(visible=True)
	vis.add_geometry(pcd)

	if front is not None: vis.get_view_control().set_front(front)
	if lookat is not None: vis.get_view_control().set_lookat(lookat)
	if up is not None: vis.get_view_control().set_up(up)
	if zoom is not None: vis.get_view_control().set_zoom(zoom)

	vis.poll_events()

	vis.update_renderer()
	vis.run()


def ply2img(ply_path, img_base_path, front=None, lookat=None, up=None, zoom=None):
	pcd = o3d.io.read_point_cloud(ply_path)

	vis = o3d.visualization.VisualizerWithEditing()
	vis.create_window(visible=False)
	vis.add_geometry(pcd)

	if front is not None: vis.get_view_control().set_front(front)
	if lookat is not None: vis.get_view_control().set_lookat(lookat)
	if up is not None: vis.get_view_control().set_up(up)
	if zoom is not None: vis.get_view_control().set_zoom(zoom)

	vis.get_render_option().background_color = np.asarray([0, 0, 0])
	vis.poll_events()
	black = np.array(vis.capture_screen_float_buffer(do_render=True)) * 255
	black = black.astype(np.uint8)

	vis.get_render_option().background_color = np.asarray([255, 255, 255])
	vis.poll_events()
	white = np.array(vis.capture_screen_float_buffer(do_render=True)) * 255
	white = white.astype(np.uint8)

	mask = white - black

	Image.fromarray(white).save(img_base_path + "_bg.png")
	Image.fromarray(mask).save(img_base_path + "_mask.png")

# ply_path = "results/p.ply"
# img_base_path = "C:/Users/dell/Desktop/inference/0"

# show_ply(ply_path)
# ply2img(ply_path, img_base_path, front, lookat, up, zoom)
