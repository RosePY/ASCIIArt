from convolution import Convolution as conv
from kernel import Kernel as ker
from utils import *
import math

class Sampling:

	FILTER_MEDIAN = 0
	FILTER_BILINEAR_INTERPOLATION = 1
	FILTER_GAUSSIAN = 2

	@staticmethod
	def upscale2d(img, nx, ny):
		a = img.copy()
		os_x, os_y = a.shape[:2]
		r_x = int(nx / os_x)
		m_x = nx % os_x
		r_y = int(ny / os_y)
		m_y = ny % os_y
		## REPLICATING ROWS AND COLUMNS UNIFORMLY
		# rows
		if int(r_x) > 1:
			a = np.repeat(a, int(r_x), axis=0).reshape(os_x * r_x, os_y)
		# columns
		if int(r_y) > 1:
			a = np.repeat(a, int(r_y), axis=1).reshape(os_x * r_x, os_y * r_y)
		# rows
		if m_x >= 1:
			dist_x = np.linspace(0, nx, m_x + 2)
			for i in dist_x[1:-1].astype(int):
				a = np.insert(a, [int(i)], a[int(i - 1)], axis=0)
		# colums
		if m_y >= 1:
			dist_y = np.linspace(0, ny, m_y + 2)

			for i in dist_y[1:-1].astype(int):
				a = np.insert(a, [int(i)], a[:, [int(i - 1)]], axis=1)

		return a

	@staticmethod
	def downscale2d(img, nx, ny):
		a = img.copy()
		os_x, os_y = a.shape[:2]
		num_x = os_x - nx
		num_y = os_y - ny
		dist_x = np.linspace(0, nx, num_x + 2)
		dist_y = np.linspace(0, ny, num_y + 2)
		dist_x = dist_x[1:-1].astype(int)
		dist_y = dist_y[1:-1].astype(int)
		for i in dist_x:
			a = np.delete(a, i, 0)
		for j in dist_y:
			a = np.delete(a, j, 1)
		return a

	def simple_resize(img, n_rows, n_cols):
		##----CAUTION---
		## THIS FUNCTION DOES NOT UPSCALE ON ONE AXIS AND DOWNSCALE IN THE OTHER
		# smooth_type = gaussian, bilinear interpolation, median or whatever
		# flag: 0 down else up
		if n_rows==img.shape[0]:
			return img
		if n_cols==img.shape[1]:
			return img
		flag = n_rows > img.shape[0]
		if len(img.shape) == 2:
			new_img = np.zeros((n_rows, n_cols))
			new_img[:, :] = Sampling.upscale2d(img[:, :], n_rows, n_cols) if flag else Sampling.downscale2d(img[:, :], n_rows, n_cols)

		else:
			new_img = np.zeros((n_rows, n_cols, img.shape[2]), np.float32)
			for i in range(img.shape[2]):
				new_img[:, :, i] = Sampling.upscale2d(img[:, :, i], n_rows, n_cols) if flag else Sampling.downscale2d(img[:, :, i], n_rows, n_cols)
		return new_img


	def smooth_resize(img, n_rows, n_cols, smooth_type):
		##----CAUTION---
		## THIS FUNCTION DOES NOT UPSCALE ON ONE AXIS AND DOWNSCALE IN THE OTHER
		# smooth_type= gaussian, bilinear interpolation, median or whatever
		# flag: 0 down else up
		if n_rows==img.shape[0]:
			return img
		if n_cols==img.shape[1]:
			return img
		flag = n_rows > img.shape[0]
		ratio_rows = (n_rows)*1.0 / img.shape[0]
		ratio_cols = (n_cols)*1.0 / img.shape[1]

		if flag:
			kernel_size = (math.ceil((ratio_rows + ratio_cols) / 2))*2-1 
		else:
			kernel_size = (math.ceil(math.log(math.ceil((math.pow(ratio_rows,-1) + math.pow(ratio_cols,-1)) / 2),2)))* 2 + 1
		resized_img = Sampling.simple_resize(img, n_rows, n_cols) if flag else Sampling.simple_resize(img, n_rows, n_cols)

		if smooth_type == Sampling.FILTER_MEDIAN:
			return conv.median_filter_nd(resized_img, kernel_size)
		elif smooth_type == Sampling.FILTER_BILINEAR_INTERPOLATION:
			return conv.convolution_nd(resized_img, ker.bilinear_interpolation(kernel_size))
		elif smooth_type == Sampling.FILTER_GAUSSIAN:
			return conv.convolution_nd(resized_img, ker.gaussian(kernel_size))
