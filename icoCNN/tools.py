"""
	Some functions that could be useful when dealing with icosahedral signals, but whose implementation is not optimal
	and shouldn't be used inside your training or inference loops

	Author: David Diaz-Guerra
	Date creation: 03/2022
	Python Version: 3.8
	Pytorch Version: 1.8.1
"""


import random
import numpy as np
from scipy.spatial.distance import cdist
import torch
import einops

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from icoCNN import icosahedral_grid_coordinates

__all__ = ["clean_vertices", "smooth_vertices", "random_icosahedral_rotation_matrix", "rotate_signal"]


def clean_vertices(x):
	"""  Turn into 0 the vertices of an icosahedral signal
	The class icoCNN.CleanVertices provides a more efficient implementation than this function if it will be used
	several times with maps of the same resolution.

	Parameters
	----------
	x : torch tensor with shape [..., 5, 2^r, 2^(r+1)]
		input icosahedral signal

	Returns
	-------
	torch tensor with shape [..., 5, 2^r, 2^(r+1)]
	"""
	r = int(np.log2(x.shape[-2]))
	template = torch.ones(x.shape[-2:], device=x.device)
	template[0,0] = 0
	template[0,2**r] = 0
	x = x * template
	return x


def smooth_vertices(x):
	"""  Replace the vertices of an icosahedral signal with the mean of their neighbours
	The class icoCNN.SmoothVertices provides a more efficient implementation than this function if it will be used
	several times with maps of the same resolution.

	Parameters
	----------
	x : torch tensor with shape [..., 5, 2^r, 2^(r+1)]
		Input icosahedral signal

	Returns
	-------
	torch tensor with shape [..., 5, 2^r, 2^(r+1)]
	"""
	r = int(np.log2(x.shape[-2]))
	x = clean_vertices(x)
	x[..., 0, 0] += einops.reduce(x[..., 1, 0] +
								  x[..., 1, 1] +
								  x[..., 0, 1] +
								  x[..., -1, 2**r].roll(1, -1) +
								  x[..., -1, 2**r - 1].roll(1, -1),
								  '... R charts -> ... 1 charts', 'mean') / 5
	x[..., 0, 2**r] += einops.reduce(x[..., 1, 2**r] +
									 x[..., 1, 2**r + 1] +
									 x[..., 0, 2**r + 1] +
									 x[..., -1, -1].roll(1, -1) +
									 x[..., 0, 2**r - 1],
									 '... R charts -> ... 1 charts', 'mean') / 5
	return x


def random_icosahedral_rotation_matrix(idx=None):
	"""  Random rotation matrix taken from the 60 icosahedral symmetries

	Parameters
	----------
	idx : int (optional)
		Index of the desired rotation matrix, following the same order as in the table in
		https://en.wikipedia.org/wiki/Icosahedral_symmetry#Isomorphism_of_I_with_A5
		None (default) takes a random matrix

	Returns
	-------
	3x3 ndarray
		Rotation matrix, you can apply it to an icosahedral signal with icoCNN.tools.rotate_signal(x, rotation_matrix)
		or to an icosahedral grid with np.matmul(ico_grid, rotation_matrix.transpose())
	"""
	if idx is None: idx = random.randrange(60)
	with pkg_resources.path('icoCNN', 'rotation_matrices.npy') as path:
		return np.load(path)[idx]


def rotate_signal(x, rotation_matrix, original_grid=None):
	"""  Rotate an icosahedral with a given rotation matrix

	Parameters
	----------
	x : torch tensor with shape [..., 5, 2^r, 2^(r+1)]
		Input icosahedral signal
	rotation_matrix : 3x3 ndarray
		It can be obtained with icoCNN.tools.random_icosahedral_rotation_matrix()
	original_grid : 4D ndarray with shape [5, 2^r, 2^(r+1), 3] (optional)
		3D Cartesian coordinates of every point of the icosahedral grid where x is defined.
		If it is not provided, it is computed inferring its resolution from the shape of x.

	Returns
	-------
	torch tensor with shape [..., 5, 2^r, 2^(r+1)]
		Rotated icosahedral signal
	"""
	if original_grid is None:
		r = int(np.log2(x.shape[-2]))
		original_grid = icosahedral_grid_coordinates(r)
	else:
		assert x.shape[-3:] == original_grid.shape[:-1]
	rotated_grid = np.matmul(original_grid, rotation_matrix.transpose())

	original_grid, rotated_grid = original_grid.reshape((-1, 3)), rotated_grid.reshape((-1, 3))
	D = cdist(original_grid, rotated_grid)
	reordered_indexes = D.argmin(0)

	original_shape = x.shape
	x = x.reshape((-1, np.prod(x.shape[-3:])))
	rotated_map = x[:, reordered_indexes].reshape(original_shape)

	return smooth_vertices(rotated_map)

