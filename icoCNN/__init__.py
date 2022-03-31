"""
	Python package with a pytorch implementation of the icosahedral CNNs (Cohen et al., 2019)

	Author: David Diaz-Guerra
	Date creation: 03/2022
	Python Version: 3.8
	Pytorch Version: 1.8.1
"""


from .icoCNN import *
from .icoGrid import icosahedral_grid_coordinates

import icoCNN.tools
import icoCNN.plots
