"""
	Plotting functions for icosahedral signals.

	Author: David Diaz-Guerra
	Date creation: 03/2022
	Python Version: 3.8
	Pytorch Version: 1.8.1
"""


import torch
import einops
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from math import log2

try:
    import vpython
except ImportError:
    vpython = None

from icoCNN import PadIco
from icoCNN.icoGrid import icosahedral_grid_coordinates, _sph2car

__all__ = ["icosahedral_scatter", "icosahedral_charts", "SphProjector", "draw_icosahedron"]


def icosahedral_scatter(x, grid=None, ax=None, cmap='inferno'):
	"""  Plot an icosahedral signal as a 3D scatter plot where each hexagonal pixel is a 3D point with the proper color
	The visual result isn't very good, but it's easy to do, so it can be useful for some quick tests.
	icoCNN.plots.draw_icosahedron provides better 3D plots using VPython.

	Parameters
	----------
	x : torch tensor or numpy ndarray with shape [5, 2^r, 2^(r+1)]
		Icosahedral signal to plot
	grid : 4D ndarray with shape [5, 2^r, 2^(r+1), 3] (optional)
		3D Cartesian coordinates of every point of the icosahedral grid where x is defined.
		If it is not provided, it is computed inferring its resolution from the shape of x.
	ax : matplotlib axes (optional)
		Axes to plot the signal in. It should have projection='3d'.
		If it is not provided, a new matplotlib figure is created.
	cmap : matplotlib cmap (optional)
		Colormap to use in the plot. Default is 'inferno'.
	"""
	if isinstance(x, torch.Tensor):
		x = x.detach().cpu().numpy()
	if grid is None:
		r = int(np.log2(x.shape[-2]))
		grid = icosahedral_grid_coordinates(r)
	else:
		assert x.shape[-3:] == grid.shape[:-1]
	if ax is None:
		ax = plt.figure().add_subplot(projection='3d')
	ax.scatter(grid[..., 0].ravel(), grid[..., 1].ravel(), grid[..., 2].ravel(), c=x.ravel(), cmap=cmap)


def icosahedral_charts(x, ax=None, cmap='inferno', colorbar=False):
	"""  Plot the 5 charts of an icosahedral signal as a 5*2^r x 2^(r+1) image
	The image is hard to interpret visually, but it's the projection used to perform the convolutions.

	Parameters
	----------
	x : torch tensor or numpy ndarray with shape [5, 2^r, 2^(r+1)] or [5, 2^r+2, 2^(r+1)+2] (padded signals)
		Icosahedral signal to plot
	ax : matplotlib axes (optional)
		Axes to plot the signal in. It can be useful to make subplots.
		If it is not provided, a new matplotlib figure is created.
	cmap : matplotlib cmap (optional)
		Colormap to use in the plot. Default is 'inferno'.
	colorbar : bool (optional)
		Include a colorbar into the plot, default is False.
	"""
	if isinstance(x, torch.Tensor):
		x = x.detach().cpu().numpy()
	if ax is None: ax = plt.figure().gca()
	C, H, W = x.shape
	assert C == 5
	im = ax.imshow(x.reshape(5 * H, W), origin='lower', cmap=cmap)
	if colorbar: plt.colorbar(im)


class SphProjector:
	"""  Class plot icosahedral signals as spherical equiangular projections
	This 2D figure is easier to interpret, but preparing the projection is a bit slow.
	After having created the object, you can use its plot_projection method to plot signals.

	Parameters
	----------
	r : int
		Resolution of the icosahedral signals that will be projected
	res_theta : int (optional)
		Number of elevation points of the projections. Default is 512
	res_phi : int (optional)
		Number of azimuth points of the projections. Default is 1024
	"""
	def __init__(self, r, res_theta=512, res_phi=1024):
		ico_grid = icosahedral_grid_coordinates(r)
		ico_grid_flat = ico_grid.reshape((-1,3))
		ico_grid_flat = np.concatenate((ico_grid_flat, np.array([[0,0,1],[0,0,-1]])))  # North and south poles

		theta = np.linspace(0, np.pi, res_theta)
		phi = np.linspace(-np.pi, np.pi, res_phi + 1)
		az, el = np.meshgrid(phi, theta)
		sph_grid = _sph2car(el, az).transpose((1,2,0))
		sph_grid_flatview = sph_grid.reshape((-1, 3))

		self.sph_proj_idxes = np.empty((res_theta, res_phi + 1), dtype=int)
		sph_proj_flatview = self.sph_proj_idxes.reshape((-1))
		for idx in range(sph_grid_flatview.shape[0]):
			sph_proj_flatview[idx] = np.argmin(cdist(sph_grid_flatview[idx, np.newaxis, :], ico_grid_flat))

	def get_projection(self, x):
		""" Compute the projection of an icosahedral signal

		Parameters
		----------
		x : torch tensor or numpy ndarray with shape [5, 2^r, 2^(r+1)]
			Icosahedral signal to plot

		Returns
		-------
		Numpy ndarray with shape res_theta x res_phi+1
			The last column of the matrix will be equal to the first one since both represent phi = -pi = +pi
		"""
		if isinstance(x, torch.Tensor):
			x = x.detach().cpu().numpy()
		ico_map_flat = x.reshape(-1)
		ico_map_flat = np.concatenate((ico_map_flat, np.array([0, 0])))  # North and south poles
		return ico_map_flat[self.sph_proj_idxes]

	def plot_projection(self, x, ax=None, cmap='inferno', colorbar=False):
		""" Plot the projection of an icosahedral signal

		Parameters
		----------
		x : torch tensor or numpy ndarray with shape [5, 2^r, 2^(r+1)]
			Icosahedral signal to project
		ax : matplotlib axes (optional)
			Axes to plot the signal in. It can be useful to make subplots.
			If it is not provided, a new matplotlib figure is created.
		cmap : matplotlib cmap (optional)
			Colormap to use in the plot. Default is 'inferno'.
		colorbar : bool (optional)
			Include a colorbar into the plot, default is False.
		"""
		sph_proj = self.get_projection(x)
		theta = np.linspace(0, 180, sph_proj.shape[0])
		theta_step = theta[1] - theta[0]
		phi = np.linspace(-180, 180, sph_proj.shape[1])
		phi_step = phi[1] - phi[0]
		if ax is None: ax = plt.figure().gca()
		im = ax.imshow(sph_proj, cmap=cmap,
					   extent=(phi[0]-phi_step/2, phi[-1]+phi_step/2, theta[-1]+theta_step/2, theta[0]-theta_step/2))
		ax.set_xlabel('Azimuth [$^\circ$]')
		ax.set_ylabel('Polar angle [$^\circ$]')
		if colorbar: plt.colorbar(im, location='bottom')


def _draw_polygon(vertices, center=None, color=None, canvas=None):
	assert vpython is not None, "You need to install VPython to use this function"
	if center is None:
		center = torch.mean(torch.stack(vertices), dim=0)
	vertices_shifted = vertices[1:] + vertices[0:1]
	for vertex, next_vertex in zip(vertices, vertices_shifted):
		vpython.triangle(v0=vpython.vertex(pos=vpython.vec(*center), color=color, canvas=canvas),
						 v1=vpython.vertex(pos=vpython.vec(*vertex), color=color, canvas=canvas),
						 v2=vpython.vertex(pos=vpython.vec(*next_vertex), color=color, canvas=canvas))


def draw_icosahedron(x, canvas=None):
	"""  Draw an icosahedral signal in 3D using VPython
	The vertices of each hexagonal pixel is computed according to their neighbors, so they aren't accurate at the edges.
	This effect is negligible when using icosahedral signals with resolution r=3 but is visible for lower resolutions.

	Parameters
	----------
	x : torch tensor or numpy ndarray with shape [5, 2^r, 2^(r+1)]
		Icosahedral signal to draw
	canvas : VPython canvas (optional)
		Canvas to draw the signal in.
		If it is not provided, the active one will be used (or a new one will be created).
	"""
	assert vpython is not None, "You need to install VPython to use this function"
	if type(x) is not torch.Tensor:
		x = torch.from_numpy(x)
	else:
		x = torch.clone(x)

	x -= x.min()
	x /= x.max()
	cmap = matplotlib.cm.get_cmap('inferno')

	r = int(log2(x.shape[-2]))
	grid = torch.from_numpy(icosahedral_grid_coordinates(r))
	grid_padding = PadIco(r, 1, preserve_vertices=True)
	grid_padded = einops.rearrange(grid_padding(einops.rearrange(grid, 'charts H W C -> C 1 charts H W', charts=5, C=3)),
								   'C 1 charts H W -> charts H W C', charts=5, C=3)
	grid_padded[:, -1, 1, :] = torch.Tensor([0.0, 0.0, +1.0])
	grid_padded[:, 1, -1, :] = torch.Tensor([0.0, 0.0, -1.0])
	Hpad = grid_padded.shape[1]
	Wpad = grid_padded.shape[2]

	for c in range(5):
		for h in range(1, Hpad - 1):
			for w in range(1, Wpad - 1):
				if not (h == 1 and (w == 1 or w == grid_padded.shape[2] // 2)):  # Hexagonal pixels
					_draw_polygon(
						[(grid_padded[c, h, w, :] + grid_padded[c, h + 1, w, :] + grid_padded[c, h + 1, w + 1, :]) / 3,
						 (grid_padded[c, h, w, :] + grid_padded[c, h + 1, w + 1, :] + grid_padded[c, h, w + 1, :]) / 3,
						 (grid_padded[c, h, w, :] + grid_padded[c, h, w + 1, :] + grid_padded[c, h - 1, w, :]) / 3,
						 (grid_padded[c, h, w, :] + grid_padded[c, h - 1, w, :] + grid_padded[c, h - 1, w - 1, :]) / 3,
						 (grid_padded[c, h, w, :] + grid_padded[c, h - 1, w - 1, :] + grid_padded[c, h, w - 1, :]) / 3,
						 (grid_padded[c, h, w, :] + grid_padded[c, h, w - 1, :] + grid_padded[c, h + 1, w, :]) / 3],
						center=grid_padded[c, h, w, :], color=vpython.vec(*cmap(x[c, h - 1, w - 1].item())[0:3]),
						canvas=canvas)
				else:  # Pentagonal vertices
					_draw_polygon(
						[(grid_padded[c, h, w, :] + grid_padded[c, h + 1, w, :] + grid_padded[c, h + 1, w + 1, :]) / 3,
						 (grid_padded[c, h, w, :] + grid_padded[c, h + 1, w + 1, :] + grid_padded[c, h, w + 1, :]) / 3,
						 (grid_padded[c, h, w, :] + grid_padded[c, h, w + 1, :] + grid_padded[c, h - 1, w, :]) / 3,
						 (grid_padded[c, h, w, :] + grid_padded[c, h - 1, w, :] + grid_padded[c, h, w - 1, :]) / 3,
						 (grid_padded[c, h, w, :] + grid_padded[c, h, w - 1, :] + grid_padded[c, h + 1, w, :]) / 3],
						center=grid_padded[c, h, w, :], color=vpython.color.gray(0.2), canvas=canvas)
		# Edges:
		vpython.cylinder(pos=vpython.vec(*grid_padded[c, 1, 1, :]),
						 axis=vpython.vec(*(grid_padded[c, -1, 1, :] - grid_padded[c, 1, 1, :])),
						 radius=0.01, color=vpython.color.gray(0.3), canvas=canvas)
		vpython.cylinder(pos=vpython.vec(*grid_padded[c, 1, 1, :]),
						 axis=vpython.vec(*(grid_padded[c, 1, Wpad // 2, :] - grid_padded[c, 1, 1, :])),
						 radius=0.01, color=vpython.color.gray(0.3), canvas=canvas)
		vpython.cylinder(pos=vpython.vec(*grid_padded[c, 1, 1, :]),
						 axis=vpython.vec(*(grid_padded[c, -1, Wpad // 2, :] - grid_padded[c, 1, 1, :])),
						 radius=0.01, color=vpython.color.gray(0.3), canvas=canvas)
		vpython.cylinder(pos=vpython.vec(*grid_padded[c, 1, Wpad // 2, :]),
						 axis=vpython.vec(*(grid_padded[c, -1, Wpad // 2, :] - grid_padded[c, 1, Wpad // 2, :])),
						 radius=0.01, color=vpython.color.gray(0.3), canvas=canvas)
		vpython.cylinder(pos=vpython.vec(*grid_padded[c, 1, Wpad // 2, :]),
						 axis=vpython.vec(*(grid_padded[c, 1, -1, :] - grid_padded[c, 1, Wpad // 2, :])),
						 radius=0.01, color=vpython.color.gray(0.3), canvas=canvas)
		vpython.cylinder(pos=vpython.vec(*grid_padded[c, 1, Wpad // 2, :]),
						 axis=vpython.vec(*(grid_padded[c, -1, -1, :] - grid_padded[c, 1, Wpad // 2, :])),
						 radius=0.01, color=vpython.color.gray(0.3), canvas=canvas)
	# North Pole vertex:
	_draw_polygon([(torch.tensor([0, 0, 1]) + grid_padded[0, -2, 1, :] + grid_padded[1, -2, 1, :]) / 3,
				   (torch.tensor([0, 0, 1]) + grid_padded[1, -2, 1, :] + grid_padded[2, -2, 1, :]) / 3,
				   (torch.tensor([0, 0, 1]) + grid_padded[2, -2, 1, :] + grid_padded[3, -2, 1, :]) / 3,
				   (torch.tensor([0, 0, 1]) + grid_padded[3, -2, 1, :] + grid_padded[4, -2, 1, :]) / 3,
				   (torch.tensor([0, 0, 1]) + grid_padded[4, -2, 1, :] + grid_padded[0, -2, 1, :]) / 3],
				  center=torch.tensor([0, 0, 1]), color=vpython.color.gray(0.3), canvas=canvas)
	# South Pole vertex:
	_draw_polygon([(torch.tensor([0, 0, -1]) + grid_padded[0, 1, -2, :] + grid_padded[1, 1, -2, :]) / 3,
				   (torch.tensor([0, 0, -1]) + grid_padded[1, 1, -2, :] + grid_padded[2, 1, -2, :]) / 3,
				   (torch.tensor([0, 0, -1]) + grid_padded[2, 1, -2, :] + grid_padded[3, 1, -2, :]) / 3,
				   (torch.tensor([0, 0, -1]) + grid_padded[3, 1, -2, :] + grid_padded[4, 1, -2, :]) / 3,
				   (torch.tensor([0, 0, -1]) + grid_padded[4, 1, -2, :] + grid_padded[0, 1, -2, :]) / 3],
				  center=torch.tensor([0, 0, -1]), color=vpython.color.gray(0.3), canvas=canvas)
