"""
	Functions to build the icosahedral grid where the icosahedral signals are defined

	Author: David Diaz-Guerra
	Date creation: 03/2022
	Python Version: 3.8
	Pytorch Version: 1.8.1
"""


import numpy as np

__all__ = ["icosahedral_grid_coordinates"]


def _sph2car(el, az, r=1):
	return np.array([
		r * np.cos(az) * np.sin(el),
		r * np.sin(az) * np.sin(el),
		r * np.cos(el)
	])


def _get_affine_transform(p, q):
	P = np.column_stack([p[1,:]-p[0,:],
						 p[2,:]-p[0,:],
						 np.cross(p[1,:]-p[0,:], p[2,:]-p[0,:])])
	Q = np.column_stack([q[1,:]-q[0,:],
						 q[2,:]-q[0,:],
						 np.cross(q[1,:]-q[0,:], q[2,:]-q[0,:])])

	R = Q.dot(np.linalg.inv(P)).transpose()
	t = q[0,:] - np.dot(p[0,:], R)

	return R, t


def _apply_affine_transform(R, t, v):
	return np.dot(v, R) + t


def _icosahedral_affine_transformations(r):
	vp = [[-1/2 + n/2, (n+1)%2 * np.sin(np.pi/3), 0] for n in range(6)]
	vp = 2**r * np.array(vp)

	vi = np.array([_sph2car(0, 0),
				   _sph2car(np.pi / 2 - np.arctan(1 / 2), 0),
				   _sph2car(np.pi / 2 - np.arctan(1 / 2), 2 * 2 * np.pi / 10),
				   _sph2car(np.pi / 2 + np.arctan(1 / 2), 1 * 2 * np.pi / 10),
				   _sph2car(np.pi / 2 + np.arctan(1 / 2), 3 * 2 * np.pi / 10),
				   _sph2car(np.pi, 0)])

	rotate_chart_matrix = np.array([[+np.cos(2*np.pi/5), +np.sin(2*np.pi/5), 0],
									[-np.sin(2*np.pi/5), +np.cos(2*np.pi/5), 0],
									[0, 0, 1]])

	R = np.empty((5, 4, 3, 3))
	T = np.empty((5, 4, 3))

	for c in range(5):
		for f in range(4):
			R[c,f,...], T[c,f,:] = _get_affine_transform(vp[f:f + 3, :], vi[f:f + 3, :])
		vi = np.dot(vi, rotate_chart_matrix)

	return R, T


def icosahedral_grid_coordinates(r):
	"""  3D Cartesian coordinates of the icosahedral grid of resolution r
	The implementation of this function is not optimal, so you shouldn't use it inside your training or inference loops.

	Parameters
	----------
	r : grid resolution

	Returns
	-------
	4D ndarray with shape [5, 2^r, 2^(r+1), 3]
		3D Cartesian coordinates of every point of the 5 x 2^r x 2^(r+1) icosahedral grid
	"""

	H = 2**r
	W = 2**(r+1)

	R, T = _icosahedral_affine_transformations(r)
	icosahedral_grid = np.empty((5,H,W,3))

	for c in range(5):
		for h in range(H):
			for w in range(W):
				xp = w - h * np.cos(np.pi/3)
				yp = h * np.sin(np.pi/3)
				f = 0 if w < 2**r and h > w \
					else 1 if w < 2**r \
					else 2 if h > w - 2**r\
					else 3
				icosahedral_grid[c,h,w] = _apply_affine_transform(R[c, f, ...], T[c, f, :], [xp, yp, 0])

	return icosahedral_grid
