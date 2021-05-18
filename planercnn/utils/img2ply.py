import numpy as np
from PIL import Image


def img2data(image, mask, camera: [float], z: float):
	h, w = image.shape[:2]

	# [x, y]  [X, Y, Z, r, g, b, a]
	data = np.zeros((h, w, 7), dtype=np.float)

	for i in range(h):  # Y = -U
		data[i, :, 1] = -(i - h / 2) / camera[0]

	for j in range(w):  # X = V
		data[:, j, 0] = (j - w / 2) / camera[1]

	data[:, :, 2] = z

	data[:, :, 3:6] = image

	data[:, :, 6] = mask

	# alpha != 0
	data = data.reshape(-1, 7)

	data_face = data[(data[:, 6] >= 245)]

	data_points = data[(data[:, 6] <= 246)]
	data_points = data_points[(data_points[:, 6] > 0)]

	return data_face, data_points


# Function to create point cloud file
def create_output(data, filename):
	np.savetxt(filename, data, fmt='%f %f %f %d %d %d %d')
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
\n
'''
	with open(filename, 'r+') as f:
		old = f.read()
		f.seek(0)
		f.write(ply_header % dict(vert_num=len(data)))
		f.write(old)


z = -2
camera = [288., 288.]

image_path = "../results/image.png"
mask_path = "../results/1_result.png"

image = np.array(Image.open(image_path))
mask = np.array(Image.open(mask_path).convert("L"))

data_face, data_points = img2data(image, mask, camera, z)

create_output(data_face, "results/p_face.ply")
create_output(data_points, "results/p_points.ply")
