
# icoCNN

**icoCNN** is a free and open-source Pytorch implementation of the icosahedral CNNs introduced in [[1]](#references). 
We developed this package as part of our work on acoustic direction of arrival estimation [[2]](#references), so please
consider citing that paper if you find this implementation useful for your research.

- [Installation](#installation)
- [License](#license)
- [Documentation](#documentation)
  * [`icosahedral_grid_coordinates`](#icosahedral_grid_coordinates)
  * [`ConvIco`](#ConvIco)
  * [`PoolIco`](#PoolIco)
  * [`LNormIco`](#LNormIco)
  * [`PadIco`](#PadIco)
  * [`UnPoolIco`]()
  * [`CleanVertices`](#CleanVertices)
  * [`SmoothVertices`](#SmoothVertices)
  * [`tools`](#tools)
  * [`plots`](#plots)
- [References](#references)

## Installation

You can use `pip` to install **icoCNN** from our repository through `pip install  https://github.com/DavidDiazGuerra/icoCNN/zipball/master`. 
You can also clone or download our repository and run `python setup.py install`.


## License

The library is subject to AGPL-3.0 license and comes with no warranty. If you find it useful for your research work, 
please, acknowledge it to [[2]](#references).

## Documentation

### `icosahedral_grid_coordinates`
3D Cartesian coordinates of the icosahedral grid of resolution r.
The implementation of this function is not optimal, so you shouldn't use it inside your training or inference loops.
#### Parameters
* **r** : *int*, grid resolution
#### Returns
*4D ndarray with shape [5, 2^r, 2^(r+1), 3]*,
3D Cartesian coordinates of every point of the 5 x 2^r x 2^(r+1) icosahedral grid

### `ConvIco`
Pytorch icosahedral convolution layer.
#### Parameters
* **r** : *int*,
        Resolution of the input icosahedral signal
* **Cin** : *int*,
        Number of channels in the input icosahedral signal (without including the Rin orientation channels)
* **Cout** : *int*,
        Number of channels produced by the convolution without including the 6 kernel orientation channels
        (i.e. the number of kernels in the convolution)
* **Rin** : *int (1 or 6)*, 
        6 when the input signal includes the 6 kernel orientation channels or 1 if it doesn't
* **bias** : *bool (optional)*,
        If True (default), adds a learnable bias to the output
* **smooth_vertices** : *bool (optional)*,
        If false (default), the vertices of the icosahedral grid are turned into 0 as done in the original paper by
        Cohen et al. If True, the vertices are replaced by the mean of their neighbours (also equivariant).
#### Shape
* **Input** : [..., Cin, Rin, 5, 2^r, 2^(r+1)]
* **Output** : [..., Cout, 6, 5, 2^r, 2^(r+1)]

### `PoolIco`
Pytorch icosahedral pooling layer.
#### Parameters
* **r** : *int*,
        Resolution of the input icosahedral signal
* **R** : *int (1 or 6)*,
        6 when the input signal includes the 6 kernel orientation channels or 1 if it doesn't
* **function** : *pytorch reduction function (optional)*,
        Function used to compute the value of every output hexagonal pixel from the 7 closest input hexagonal pixels.
        It should be a reduction function that can be called as function(x, -1) to reduce the last dimension of x.
        Default: torch.mean
* **smooth_vertices** : *bool (optional)*,
        If false (default), the vertices of the icosahedral grid are turned into 0 as done in the original paper by
        Cohen et al. If True, the vertices are replaced by the mean of their neighbours (also equivariant).
#### Shape
* **Input** : [..., R, 5, 2^r, 2^(r+1)]
* **Output** : [..., R, 5, 2^(r-1), 2^r]

### `LNormIco`
Pytorch icosahedral layer normalization layer.
#### Parameters
* **C** : *int*,
        Number of channels in the input icosahedral signal (without including the Rin orientation channels)
* **R** : *int (1 or 6)*, 
        6 when the input signal includes the 6 kernel orientation channels or 1 if it doesn't
#### Shape
* **Input** : [..., C, R, 5, 2^r, 2^(r+1)]
* **Output** : [..., C, R, 5, 2^r, 2^(r+1)]

### `PadIco`
Pytorch module to pad every chart of an icosahedral signal.
`icoCNN.ConvIco` already incorporates this padding, so you probably don't want to directly use this class.
#### Parameters
* **r** : *int*,
        Resolution of the input icosahedral signal
* **R** : *int (1 or 6)*,
        6 when the input signal includes the 6 kernel orientation channels or 1 if it doesn't
* **smooth_vertices** : *bool (optional)*,
        If False (default), the vertices of the icosahedral grid are turned into 0 as done in the original paper by
        Cohen et al. If True, the vertices are replaced by the mean of their neighbours (also equivariant).
* **preserve_vertices** : *bool (optional)*,
        If True, it avoids turning the vertices into 0 (not equivariant). Default is False.
#### Shape
* **Input** : [..., R, 5, 2^r, 2^(r+1)]
* **Output** : [..., R, 5, 2^r+2, 2^(r+1)+2]

### `UnPoolIco`
Pytorch icosahedral unpooling layer
#### Parameters
* **r** : *int*,
        Resolution of the input icosahedral signal
* **R** : *int (1 or 6)*,
        6 when the input signal includes the 6 kernel orientation channels or 1 if it doesn't
#### Shape
* **Input** : [..., R, 5, 2^r, 2^(r+1)]
* **Output** : [..., R, 5, 2^(r+1), 2^(r+2)]

### `CleanVertices`
Pytorch layer to turn into 0 the vertices of icosahedral signals.
#### Parameters
* **r** : *int*, Resolution of the input icosahedral signal
#### Shape
* **Input** : [..., 5, 2^r, 2^(r+1)]
* **Output** : [..., 5, 2^r, 2^(r+1)]

### `SmoothVertices`
Pytorch layer to replace the vertices of icosahedral signals with the mean of their neighbours.
#### Parameters
* **r** : *int*, Resolution of the input icosahedral signal
#### Shape
* **Input** : [..., 5, 2^r, 2^(r+1)]
* **Output** : [..., 5, 2^r, 2^(r+1)]

### `tools`
This subpackage includes some functions that could be useful when dealing with icosahedral signals, but whose 
implementation is not optimal and shouldn't be used inside your training or inference loops. You can find more 
information in its [documentation file](tools.md).

### `plots`
This subpackage includes some plotting functions for icosahedral signals. You can find more information in its 
[documentation file](plots.md).

## References

[1] T. Cohen, M. Weiler, B. Kicanaoglu, and M. Welling, 
“Gauge Equivariant Convolutional Networks and the Icosahedral CNN,” 
in Proceedings of the 36th International Conference on Machine Learning. PMLR, May 2019, pp. 1321–1330
[[PMLR](http://proceedings.mlr.press/v97/cohen19d.html)][[arXiv](https://arxiv.org/abs/1902.04615)]


[2] Diaz-Guerra, D., Miguel, A. & Beltran, J.R. 
Direction of Arrival Estimation of Sound Sources Using Icosahedral CNNs.
[[arXiv preprint](https://arxiv.org/abs/2203.16940)]
