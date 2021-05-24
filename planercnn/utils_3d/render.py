import numpy as np


class Render:

	def __init__(self, k, f, image_range):
		'''
		A 3D render class, used for .ply input.
		The camera in at (0, 0, 0) in default.
		The photo plane is (0, 0, f), and the normal is (0, 0, 1)
		:param k: the size of each pixel (e.g. k = 580 means 580 pixels occupy 1m)
		:param f: the position of photo plane
		:param image_range: the output size of photo, like [[x_min, x_max], [y_min, y_max]]
		'''
		self.point_list, self.face_list = None, None
		self.k = k
		self.f = f
		self.image_range = image_range

		# output saved
		self.h, self.w = image_range[0][1] - image_range[0][0], image_range[1][1] - image_range[1][0]
		self.image = np.zeros((self.h, self.w, 3))
		self.depth = np.zeros((self.h, self.w))
		self.mask = np.zeros((self.h, self.w))

	# read .ply file
	def read_ply(self, ply_path):

		with open(ply_path, "r") as f:
			content = f.read()
			f.close()

		head, list = content.split("end_header", 1)
		head = head.split("\n")
		list = list.lstrip("\n").rstrip("\n").split("\n")

		vertex_num = 0
		for item in head:
			if item[:15] == "element vertex ": vertex_num = int(item[15:])

		vertex_list, face_list = list[:vertex_num], list[vertex_num:]

		# [x',y',z,r,g,b], where x'=-y, y'=x
		self.point_list = np.array(
			[[float(item) for item in vertex.rstrip(" ").split(" ")[:6]] for vertex in vertex_list])
		self.point_list[:, 1], self.point_list[:, 0] = self.point_list[:, 0], -self.point_list[:, 1]

		# [pid0,pid1,pid2]
		self.face_list = np.array([[int(item) for item in face.rstrip(" ").split(" ")[1:]] for face in face_list])

	# project the points into the photographing plane, and convert to pos of pixels (i, j)
	def point_projecting(self):
		# i,j are the position of pixels
		i, j = self.point_list[:, 0] / self.point_list[:, 2] * self.k * self.f - self.image_range[0][0], \
			   self.point_list[:, 1] / self.point_list[:, 2] * self.k * self.f - self.image_range[1][0]
		self.point_list = np.insert(self.point_list, 6, values=i, axis=1)
		self.point_list = np.insert(self.point_list, 7, values=j, axis=1)

	# render one face
	def render_one_face(self, face):

		# outside the image : pass
		if not self.is_face_in_range(face): return

		# find all integer pixels that are inside the triangle
		U, V = self.inside_triangle(face)
		ids = self.point_in_range(U, V)

		# not exist : skip
		if len(ids) == 0: return

		U, V = U[ids], V[ids]

		# exist inside pixels, render them
		points0, x, y, x2, y2, xy, base = self.face_compute_vectors(face)

		S = np.stack([U, V], 0).T - points0  # (len, 2)

		# compute ratio
		X0, Y0 = np.dot(S, x), np.dot(S, y)  # (len)
		A, B = (Y0 * xy - X0 * y2) / base, (X0 * xy - Y0 * x2) / base  # (len)

		# compute weight
		All = 1 - A * B  # (len)
		Weight = np.array([(1 - A) * (1 - B) / All, A * (1 - B) / All, (1 - A) * B / All]).T  # (len, 3)

		Depth = self.weight_avg(Weight, face, [2]).reshape(len(Weight))  # depth

		# bypass those previous render
		ids = np.arange(len(ids))[(self.mask[U, V] == 0) | (self.depth[U, V] < Depth)]
		U, V, Weight, Depth = U[ids], V[ids], Weight[ids], Depth[ids]

		self.depth[U, V] = Depth  # now render
		self.image[U, V] = self.weight_avg(Weight, face, [3, 4, 5])
		self.mask[U, V] = 1  # mask

	# render by all faces
	def render_by_faces(self):
		for id in range(len(self.face_list)):
			self.render_one_face(self.face_list[id])

	# judge if a point with pos i,j is in image range
	def point_in_range(self, U, V):
		return np.arange(len(U))[(U >= 0) & (U < self.h) & (V >= 0) & (V < self.w)]

	# judge if a face  is in image range
	def is_face_in_range(self, face):
		return len(self.point_in_range(self.point_list[face][:, 6], self.point_list[face][:, 7])) != 0

	# find all integer points that are inside the triangle
	def inside_triangle(self, face):

		x1, y1 = self.point_list[face[0]][6], self.point_list[face[0]][7]
		x2, y2 = self.point_list[face[1]][6], self.point_list[face[1]][7]
		x3, y3 = self.point_list[face[2]][6], self.point_list[face[2]][7]

		xs = np.array((x1, x2, x3), dtype=float)
		ys = np.array((y1, y2, y3), dtype=float)

		# The possible range of coordinates that can be returned
		x_range = np.arange(int(np.min(xs)), int(np.max(xs)) + 1)
		y_range = np.arange(int(np.min(ys)), int(np.max(ys)) + 1)

		# no points
		if len(x_range) == 0 or len(y_range) == 0: return [], []

		# Set the grid of coordinates on which the triangle lies. The centre of the
		# triangle serves as a criterion for what is inside or outside the triangle.
		X, Y = np.meshgrid(x_range, y_range)
		xc = np.mean(xs)
		yc = np.mean(ys)

		# From the array 'triangle', points that lie outside the triangle will be
		# set to 'False'.
		triangle = np.ones(X.shape, dtype=bool)
		for i in range(3):
			ii = (i + 1) % 3
			if xs[i] == xs[ii]:
				include = X * (xc - xs[i]) / abs(xc - xs[i]) >= xs[i] * (xc - xs[i]) / abs(xc - xs[i])
			else:
				poly = np.poly1d(
					[(ys[ii] - ys[i]) / (xs[ii] - xs[i]), ys[i] - xs[i] * (ys[ii] - ys[i]) / (xs[ii] - xs[i])])
				include = Y * (yc - poly(xc)) / abs(yc - poly(xc)) >= poly(X) * (yc - poly(xc)) / abs(yc - poly(xc))
			triangle *= include

		# Output: 2 arrays with the x- and y- coordinates of the points inside the
		# triangle.
		return X[triangle], Y[triangle]

	# compute for vectors of face
	def face_compute_vectors(self, face):

		# get pos of points
		point0 = np.array([self.point_list[face[0]][6], self.point_list[face[0]][7]])
		point1 = np.array([self.point_list[face[1]][6], self.point_list[face[1]][7]])
		point2 = np.array([self.point_list[face[2]][6], self.point_list[face[2]][7]])

		# get vectors
		x = point1 - point0
		y = point2 - point0

		# comput
		x2 = np.dot(x, x)
		y2 = np.dot(y, y)
		xy = np.dot(x, y)

		base = xy ** 2 - x2 * y2

		return point0, x, y, x2, y2, xy, base

	# compute faces average with weight
	def weight_avg(self, Weight, face, value_id: [float]):
		return np.dot(Weight, self.point_list[face][:, value_id])

	# render one pixel
	def render_one_pixel_by_point(self, u, v, th=0.1, e=1e-2, gamma=0.1):

		# find all list
		list = self.point_list[self.split_points_set[u][v]]

		# bypass all points that is too deep
		z_min = list[:, 2].min()
		list = list[list[:, 2] < z_min + th]

		# compute weight for each points
		Weight = e / (list[:, 2] - z_min + e) + gamma * (1 - abs(list[:, 6] - u) - abs(list[:, 7] - v))
		Weight /= Weight.sum()

		# compute value by weight
		self.depth[u][v] = self.weight_points_avg(Weight, list, [2])  # now render
		self.image[u][v] = self.weight_points_avg(Weight, list, [3, 4, 5])
		self.mask[u][v] = 1  # mask

	# render by points
	def render_by_point(self, th=0.1, e=1e-2, gamma=0.1):
		self.split_points_set = self.split_point_range()
		for i in range(self.h):
			for j in range(self.w):
				if self.mask[i][j] == 1 or len(self.split_points_set[i][j]) == 0: continue
				self.render_one_pixel_by_point(i, j, th=th, e=e, gamma=gamma)

	# split points by pixel range
	def split_point_range(self):
		# split
		points_range = [[[] for _ in range(self.w)] for _ in range(self.h)]
		for pid in range(len(self.point_list)):
			u, v = int(self.point_list[pid, 6].round()), int(self.point_list[pid, 7].round())
			if u < 0 or u >= self.h or v < 0 or v >= self.w: continue
			points_range[u][v].append(pid)
		return points_range

	# compute points average with weight
	def weight_points_avg(self, Weight, points, value_id: [float]):
		return np.dot(Weight, points[:, value_id])

	# render by point quickly
	def quick_render(self, point_size=3):
		size = point_size // 2

		# get U and V
		U, V = (self.point_list[:, 0] / self.point_list[:, 2] * self.k * self.f - self.image_range[0][0]). \
				   round().astype(np.int), \
			   (self.point_list[:, 1] / self.point_list[:, 2] * self.k * self.f - self.image_range[1][0]). \
				   round().astype(np.int)

		# remain those in the camera range
		in_range = (U >= 0) & (U < self.h) & (V >= 0) & (V < self.w)
		U, V, self.point_list = U[in_range], V[in_range], self.point_list[in_range]

		# depth and rgb
		Depth, RGB = self.point_list[:, 2], self.point_list[:, 3:6]

		# range of points affect
		U_min, U_max, V_min, V_max = U.copy() - size, U.copy() + size + 1, V.copy() - size, V.copy() + size + 1
		U_min[U_min < 0] = 0
		U_max[U_max >= self.h] = self.h
		V_min[V_min < 0] = 0
		V_max[V_max >= self.w] = self.w

		# render
		for pid in range(len(self.point_list)):
			u, v, depth = U[pid], V[pid], Depth[pid]
			u_min, u_max, v_min, v_max = U_min[pid], U_max[pid], V_min[pid], V_max[pid]
			render_range = (self.mask[u_min:u_max, v_min:v_max] == 0) | (self.depth[u_min:u_max, v_min:v_max] < depth)
			self.image[u_min:u_max, v_min:v_max][render_range] = RGB[pid]
			self.depth[u_min:u_max, v_min:v_max][render_range] = depth
			self.mask[u_min:u_max, v_min:v_max][render_range] = 1
