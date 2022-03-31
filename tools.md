
## `icoCNN.tools`

This subpackage includes some functions that could be useful when dealing with icosahedral signals, but whose 
implementation is not optimal and shouldn't be used inside your training or inference loops.

 - [`clean_vertices`](#clean_vertices)
 - [`smooth_vertices`](#smooth_vertices)
 - [`random_icosahedral_rotation_matrix`](#random_icosahedral_rotation_matrix)
 - [`rotate_signal`](#rotate_signal)

### `clean_vertices`
Turn into 0 the vertices of an icosahedral signal.
	The class `icoCNN.CleanVertices` provides a more efficient implementation than this function if it will be used
	several times with maps of the same resolution.
#### Parameters
* **x** : *torch tensor with shape [..., 5, 2^r, 2^(r+1)]*, 
		input icosahedral signal
#### Returns
*torch tensor with shape [..., 5, 2^r, 2^(r+1)]*

### `smooth_vertices`
Replace the vertices of an icosahedral signal with the mean of their neighbours.
	The class `icoCNN.SmoothVertices` provides a more efficient implementation than this function if it will be used
	several times with maps of the same resolution.
#### Parameters
* **x** : *torch tensor with shape [..., 5, 2^r, 2^(r+1)]*, 
		input icosahedral signal
#### Returns
*torch tensor with shape [..., 5, 2^r, 2^(r+1)]*

### `random_icosahedral_rotation_matrix`
Random rotation matrix taken from the 60 icosahedral symmetries.
#### Parameters
* **idx** : *int (optional)*, 
		Index of the desired rotation matrix, following the same order as in [this table](https://en.wikipedia.org/wiki/Icosahedral_symmetry#Isomorphism_of_I_with_A5).
		None (default) takes a random matrix
#### Returns
*3x3 ndarray*,
		Rotation matrix, you can apply it to an icosahedral signal with `icoCNN.tools.rotate_signal(x, rotation_matrix)`
		or to an icosahedral grid with `np.matmul(ico_grid, rotation_matrix.transpose())`

### `rotate_signal`
Rotate an icosahedral with a given rotation matrix.
#### Parameters
* **x** : *torch tensor with shape [..., 5, 2^r, 2^(r+1)]*, 
        Input icosahedral signal
* **rotation_matrix** : *3x3 ndarray*
        It can be obtained with icoCNN.tools.random_icosahedral_rotation_matrix()
* **original_grid** : *4D ndarray with shape [5, 2^r, 2^(r+1), 3] (optional)*,
        3D Cartesian coordinates of every point of the icosahedral grid where x is defined.
        If it is not provided, it is computed inferring its resolution from the shape of x.
#### Returns
*torch tensor with shape [..., 5, 2^r, 2^(r+1)]*,
		Rotated icosahedral signal
