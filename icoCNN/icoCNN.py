"""
	Pytorch modules to build icosahedral CNNs

	Author: David Diaz-Guerra
	Date creation: 03/2022
	Python Version: 3.8
	Pytorch Version: 1.8.1
"""


import torch
import einops
from math import sqrt

__all__ = ["CleanVertices", "SmoothVertices", "PadIco", "ConvIco", "PoolIco", "LNormIco"]


class CleanVertices(torch.nn.Module):
	"""  Pytorch layer to turn into 0 the vertices of icosahedral signals

	Parameters
	----------
	r : Resolution of the input icosahedral signal

	Shape
	-----
	Input : [..., 5, 2^r, 2^(r+1)]
	Output : [..., 5, 2^r, 2^(r+1)]
	"""
	def __init__(self, r):
		super().__init__()
		self.register_buffer('mask', torch.ones((2**r, 2**(r+1))))
		self.mask[0, 0] = 0
		self.mask[0, 2**r] = 0

	def forward(self, x):
		return x * self.mask


class SmoothVertices(torch.nn.Module):
	"""  Pytorch layer to replace the vertices of icosahedral signals with the mean of their neighbours

	Parameters
	----------
	r : Resolution of the input icosahedral signal

	Shape
	-----
	Input : [..., 5, 2^r, 2^(r+1)]
	Output : [..., 5, 2^r, 2^(r+1)]
	"""
	def __init__(self, r):
		super().__init__()
		self.r = r
		self.clear_vertices = CleanVertices(r)
		self.v1_neighbors = torch.LongTensor([[[chart, 1, 0],
											   [chart, 1, 1],
											   [chart, 0, 1],
											   [chart-1, -1, 2**r],
											   [chart-1, -1, 2**r-1]] for chart in range(5)])
		self.v2_neighbors = torch.LongTensor([[[chart, 1, 2**r],
											   [chart, 1, 2**r + 1],
											   [chart, 0, 2**r + 1],
											   [chart-1, -1, -1],
											   [chart, 0, 2**r-1]] for chart in range(5)])

	def forward(self, x):
		x = self.clear_vertices(x)
		x[..., 0, 0] += einops.reduce(x[...,
										self.v1_neighbors[..., 0],
										self.v1_neighbors[..., 1],
										self.v1_neighbors[..., 2]],
									  '... R charts neighbors -> ... 1 charts', 'mean')
		x[..., 0, 2**self.r] += einops.reduce(x[...,
												self.v2_neighbors[..., 0],
												self.v2_neighbors[..., 1],
												self.v2_neighbors[..., 2]],
											  '... R charts neighbors -> ... 1 charts', 'mean')
		return x


class PadIco(torch.nn.Module):
	"""  Pytorch module to pad every chart of an icosahedral signal
	icoCNN.ConvIco already incorporates this padding, so you probably don't want to directly use this class.

	Parameters
	----------
	r : int
		Resolution of the input icosahedral signal
	R : int, 1 or 6
		6 when the input signal includes the 6 kernel orientation channels or 1 if it doesn't
	smooth_vertices : bool (optional)
		If False (default), the vertices of the icosahedral grid are turned into 0 as done in the original paper by
		Cohen et al. If True, the vertices are replaced by the mean of their neighbours (also equivariant).
	preserve_vertices : bool (optional)
		If True, it avoids turning the vertices into 0 (not equivariant). Default is False.

	Shape
	-----
	Input : [..., R, 5, 2^r, 2^(r+1)]
	Output : [..., R, 5, 2^r+2, 2^(r+1)+2]
	"""
	def __init__(self, r, R, smooth_vertices=False, preserve_vertices=False):
		super().__init__()
		assert R==1 or R==6
		self.R = R
		self.r = r
		self.H = 2**r
		self.W = 2**(r+1)

		self.smooth_vertices = smooth_vertices
		if not preserve_vertices:
			self.process_vertices = SmoothVertices(r) if smooth_vertices else CleanVertices(r)
		else:
			assert not smooth_vertices
			self.process_vertices = lambda x: x

		idx_in= torch.arange(R * 5 * self.H * self.W, dtype=torch.long).reshape(R, 5, self.H, self.W)
		idx_out = torch.zeros((R, 5, self.H + 2, self.W + 2), dtype=torch.long)
		idx_out[..., 1:-1, 1:-1] = idx_in
		idx_out[..., 0, 1:2 ** r + 1] = idx_in.roll(1, -3)[..., -1, 2 ** r:]
		idx_out[..., 0, 2 ** r + 1:-1] = idx_in.roll(1, -3).roll(-1, -4)[..., :, -1].flip(-1)
		idx_out[..., -1, 2:2 ** r + 2] = idx_in.roll(-1, -3).roll(-1, -4)[..., :, 0].flip(-1)
		idx_out[..., -1, 2 ** r + 1:-1] = idx_in.roll(-1, -3)[..., 0, 0:2 ** r]
		idx_out[..., 1:-1, 0] = idx_in.roll(1, -3).roll(1, -4)[..., -1, 0:2 ** r].flip(-1)
		idx_out[..., 2:, -1] = idx_in.roll(-1, -3).roll(1, -4)[..., 0, 2 ** r:].flip(-1)
		self.reorder_idx = idx_out

	def forward(self, x):
		x = self.process_vertices(x)
		if self.smooth_vertices:
			smooth_north_pole = einops.reduce(x[..., -1, 0], '... R charts -> ... 1 1', 'mean')
			smooth_south_pole = einops.reduce(x[..., 0, -1], '... R charts -> ... 1 1', 'mean')

		x = einops.rearrange(x, '... R charts H W -> ... (R charts H W)', R=self.R, charts=5, H=self.H, W=self.W)
		y = x[..., self.reorder_idx]

		if self.smooth_vertices:
			y[..., -1, 1] = smooth_north_pole
			y[..., 1, -1] = smooth_south_pole

		return y


class ConvIco(torch.nn.Module):
	"""  Pytorch icosahedral convolution layer

	Parameters
	----------
	r : int
		Resolution of the input icosahedral signal
	Cin : int
		Number of channels in the input icosahedral signal (without including the Rin orientation channels)
	Cout : int
		Number of channels produced by the convolution without including the 6 kernel orientation channels
		(i.e. the number of kernels in the convolution)
	Rin : int, 1 or 6
		6 when the input signal includes the 6 kernel orientation channels or 1 if it doesn't
	bias : bool (optional)
		If True (default), adds a learnable bias to the output
	smooth_vertices : bool (optional)
		If False (default), the vertices of the icosahedral grid are turned into 0 as done in the original paper by
		Cohen et al. If True, the vertices are replaced by the mean of their neighbours (also equivariant).

	Shape
	-----
	Input : [..., Cin, Rin, 5, 2^r, 2^(r+1)]
	Output : [..., Cout, 6, 5, 2^r, 2^(r+1)]
	"""
	def __init__(self, r, Cin, Cout, Rin, Rout=6, bias=True, smooth_vertices=False):
		super().__init__()
		assert Rin == 1 or Rin == 6
		self.r = r
		self.Cin = Cin
		self.Cout = Cout
		self.Rin = Rin
		self.Rout = Rout

		self.process_vertices = SmoothVertices(r) if smooth_vertices else CleanVertices(r)
		self.padding = PadIco(r, Rin, smooth_vertices=smooth_vertices)

		s = sqrt(2 / (3 * 3 * Cin * Rin))
		self.weight = torch.nn.Parameter(s * torch.randn((Cout, Cin, Rin, 7)))  # s * torch.randn((Cout, Cin, Rin, 7))  #
		if bias:
			self.bias = torch.nn.Parameter(torch.zeros(Cout))
		else:
			self.register_parameter('bias', None)

		self.kernel_expansion_idx = torch.zeros((Cout, Rout, Cin, Rin, 9, 4), dtype=int)
		self.kernel_expansion_idx[..., 0] = torch.arange(Cout).reshape((Cout, 1, 1, 1, 1))
		self.kernel_expansion_idx[..., 1] = torch.arange(Cin).reshape((1, 1, Cin, 1, 1))
		idx_r = torch.arange(0, Rin)
		idx_k = torch.Tensor(((5, 4, -1, 6, 0, 3, -1, 1, 2),
							  (4, 3, -1, 5, 0, 2, -1, 6, 1),
							  (3, 2, -1, 4, 0, 1, -1, 5, 6),
							  (2, 1, -1, 3, 0, 6, -1, 4, 5),
							  (1, 6, -1, 2, 0, 5, -1, 3, 4),
							  (6, 5, -1, 1, 0, 4, -1, 2, 3)))
		for i in range(Rout):
			self.kernel_expansion_idx[:, i, :, :, :, 2] = idx_r.reshape((1, 1, Rin, 1))
			self.kernel_expansion_idx[:, i, :, :, :, 3] = idx_k[i,:]
			idx_r = idx_r.roll(1)

	def extra_repr(self):
		return "r={}, Cin={}, Cout={}, Rin={}, Rout={}, bias={}"\
			.format(self.r, self.Cin, self.Cout, self.Rin, self.Rout, self.bias is not None)

	def get_kernel(self):
		kernel = self.weight[self.kernel_expansion_idx[..., 0],
							 self.kernel_expansion_idx[..., 1],
							 self.kernel_expansion_idx[..., 2],
							 self.kernel_expansion_idx[..., 3]]
		kernel = kernel.reshape((self.Cout, self.Rout, self.Cin, self.Rin, 3, 3))
		kernel[..., 0, 2] = 0
		kernel[..., 2, 0] = 0
		return kernel

	def forward(self, x):
		x = self.padding(x)
		x = einops.rearrange(x, '... C R charts H W -> ... (C R) (charts H) W', C=self.Cin, R=self.Rin, charts=5)
		if x.ndim == 3:
			x = x.unsqueeze(0)
			remove_batch_size = True
		else:
			remove_batch_size = False
			batch_shape = x.shape[:-3]
			x = x.reshape((-1,) + x.shape[-3:])

		kernel = self.get_kernel()
		kernel = einops.rearrange(kernel, 'Cout Rout Cin Rin Hk Wk -> (Cout Rout) (Cin Rin) Hk Wk', Hk=3, Wk=3)
		bias = einops.repeat(self.bias, 'Cout -> (Cout Rout)', Cout=self.Cout, Rout=self.Rout) \
			if self.bias is not None else None

		y = torch.nn.functional.conv2d(x, kernel, bias, padding=(1, 1))
		y = einops.rearrange(y, '... (C R) (charts H) W -> ... C R charts H W', C=self.Cout, R=self.Rout, charts=5)
		y = y[..., 1:-1, 1:-1]
		if remove_batch_size: y = y[0, ...]
		else: y = y.reshape(batch_shape + y.shape[1:])

		return self.process_vertices(y)


class PoolIco(torch.nn.Module):
	"""  Pytorch icosahedral pooling layer

	Parameters
	----------
	r : int
		Resolution of the input icosahedral signal
	R : int, 1 or 6
		6 when the input signal includes the 6 kernel orientation channels or 1 if it doesn't
	function : pytorch reduction function (optional)
		Function used to compute the value of every output hexagonal pixel from the 7 closest input hexagonal pixels.
		It should be a reduction function that can be called as function(x, -1) to reduce the last dimension of x.
		Default: torch.mean
	smooth_vertices : bool (optional)
		If False (default), the vertices of the icosahedral grid are turned into 0 as done in the original paper by
		Cohen et al. If True, the vertices are replaced by the mean of their neighbours (also equivariant).

	Shape
	-----
	Input : [..., R, 5, 2^r, 2^(r+1)]
	Output : [..., R, 5, 2^(r-1), 2^r]
	"""
	def __init__(self, r, R, function=torch.mean, smooth_vertices=False):
		super().__init__()
		self.function = function
		self.padding = PadIco(r, R, smooth_vertices=smooth_vertices)
		self.process_vertices = SmoothVertices(r-1) if smooth_vertices else CleanVertices(r-1)

		self.neighbors = torch.zeros((2**(r-1), 2**r, 7, 2), dtype=torch.long)
		for h in range(self.neighbors.shape[0]):
			for w in range(self.neighbors.shape[1]):
				self.neighbors[h,w,...] = torch.Tensor([[1+2*h,   1+2*w  ],
														[1+2*h+1, 1+2*w  ],
														[1+2*h+1, 1+2*w+1],
														[1+2*h,   1+2*w+1],
														[1+2*h-1, 1+2*w  ],
														[1+2*h-1, 1+2*w-1],
														[1+2*h,   1+2*w-1]])

	def forward(self, x):
		x = self.padding(x)
		receptive_field = x[..., self.neighbors[...,0], self.neighbors[...,1]]
		y = self.function(receptive_field, -1)
		return self.process_vertices(y)

class UnPoolIco(torch.nn.Module):
	"""  Pytorch icosahedral unpooling layer

	Parameters
	----------
	r : int
		Resolution of the input icosahedral signal
	R : int, 1 or 6
		6 when the input signal includes the 6 kernel orientation channels or 1 if it doesn't


	Shape
	-----
	Input : [..., R, 5, 2^r, 2^(r+1)]
	Output : [..., R, 5, 2^(r+1), 2^(r+2)]
	"""

	def __init__(self, r, R):
		super().__init__()
		self.r = r
		self.R = R
		self.rows = 1+2*torch.arange(2^(r+1)).unsqueeze(1) # center of the 
		self.cols = 1+2*torch.arange(2^(r+2)).unsqueeze(0)
		self.padding = PadIco(r+1, R)

	def forward(self, x):
		y = torch.zeros((x.shape[:-2], 2^(self.r+1), 2^(self.r+2)))
		y = self.padding(y)
		y[..., self.rows, self.cols] = x
		y = y[..., 1:-1, 1:-1]
		return y

class LNormIco(torch.nn.Module):
	"""  Pytorch icosahedral layer normalization layer

	Parameters
	----------
	C : int
		Number of channels in the input icosahedral signal (without including the Rin orientation channels)
	R : int, 1 or 6
		6 when the input signal includes the 6 kernel orientation channels or 1 if it doesn't

	Shape
	-----
	Input : [..., C, R, 5, 2^r, 2^(r+1)]
	Output : [..., C, R, 5, 2^r, 2^(r+1)]
	"""
	def __init__(self, C, R):
		super().__init__()
		self.norm = torch.nn.LayerNorm((C, R), elementwise_affine=False)
		self.weight = torch.nn.Parameter(torch.ones((C, 1)))
		self.bias = torch.nn.Parameter(torch.zeros((C, 1)))

	def forward(self, x):
		x = einops.rearrange(x, "... C R charts H W -> ... charts H W C R")
		original_shape = x.shape
		x = einops.rearrange(x, "... charts H W C R -> (... charts H W) C R")
		x = self.norm(x)
		x = x * self.weight + self.bias
		x = x.reshape(original_shape)
		x = einops.rearrange(x, "... charts H W C R -> ... C R charts H W")
		return x
