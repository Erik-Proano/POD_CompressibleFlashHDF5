# Proper Orthogonal Decomposition for HDf5 Output Data from the FLASH code
Erik Proano 06/20/2018
___
## POD Modes Obtention
The file POD_v2.py computes the eigenvalues and eigenvectors, a.k.a POD modes, of a set of HDF5 data file extracted from the FLASH code based on the Singular Value Decomposition available from the numpy API.
Taking advantage of the yt and numpy APIs for python, this code computes the POD modes of a scalar field for later computing the time coefficients and sequential reconstruction of the most dominant modes using the snapshots method for POD.
This code is still preliminar and it still has some bugs.
## POD Module (PODMod.py)
This module contains all the necessary functions for extracting the POD modes and perform the Galerkin projection
### ReynoldsDecomp.py
This function performs a Reynolds decomposition of the simulation data. It extracts the average or mean flow. To extract the fluctuations at a desired snapshot, substract the average obtained by this function to the grid data initially provided.
### getFileName.py
This method returns 0a list containing the names of the files intended to read. If such files do not exist, then an exception returns an error indicating that the path or the file name may be incorrect.
### FlowFavreAverage
Computes Favre averages or density-weighted averages of the kinematic and thermodynamic quantities for each output file 
### Favre Average
Computed Favre averages for time series yt data structure 
### getPODmodes
Extract the eigenvectors a.k.a. POD modes using the method of snapshots
