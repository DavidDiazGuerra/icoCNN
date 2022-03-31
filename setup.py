"""
	Python package with a pytorch implementation of the icosahedral CNNs (Cohen et al., 2019)

	Author: David Diaz-Guerra
	Date creation: 03/2022
	Python Version: 3.8
	Pytorch Version: 1.8.1
"""


from setuptools import setup


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
	name='icoCNN',
	version='0.9',
	install_reqs=["numpy", "scipy", "torch", "einops", "matplotlib"],
	extras_require={
		"draw3D": ["vpython"]
	},
	packages=['icoCNN'],
	url='https://github.com/DavidDiazGuerra/icoCNN',
	license='GNU Affero General Public License v3.0',
	author='David Diaz-Guerra',
	author_email='ddga@unizar.es',
	description='Python package with a pytorch implementation of the icosahedral CNNs (Cohen et al., 2019)',
	long_description=long_description,
	long_description_content_type="text/markdown"
)
