# Proper Orthogonal Decomposition for HDf5 Output Data from the FLASH code
Erik Proano 06/20/2018
___
## POD Modes Obtention
The file POD_v1.py computes the eigenvalues and eigenvectors, a.k.a POD modes, of
a set of HDF5 data file extracted from the FLASH code.
Taking advantage of the yt and numpy APIs for python, this code computes the POD modes
of a scalar field for later computing the time coefficients and sequential reconstruction
of the most dominant modes using the snapshots method for POD.
This code is still preliminar and it still has some bugs.

