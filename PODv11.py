"""
  DESCRIPTION
    This script computes the POD modes of a Flash simulation with
    HDF5 data output corresponding to a compressible 2D simulation
    The POD modes, a.k.a. Eigenfunctions, are computed based on 
    the Singular-Value Decomposition (SVD) provided by numpy API.

  AUTHOR
    Erik S. Proano
    Embry-Riddle Aeronautical University

"""

from yt.funcs import mylog
import yt
import h5py
import numpy as np
#from mpi4py import MPI
import multiprocessing as mproc
import os
import PODMod as pod
import Flash_Post as post
import sys

def _velx_POD(field,data):
	global velx_POD
	return velx_POD

def _vely_POD(field, data):
	global vely_POD
	return vely_POD
	
def _soundsp_POD(field, data):
	global sndsp_POD
	return sndsp_POD

def _velmag_POD(field, data):
	return (data['velx_POD']**2+data['vely_POD']**2)**0.5
	
def _velocity_Fluc(field, data):
	global average
	vel_ave = yt.YTArray((average[:,0]**2-average[:,1]**2)**0.5, ('cm/s'))
	return data['velocity_magnitude']-vel_ave

def procRank(Pid, Nproc):
   if Pid == 0:
      print(location)
      print('Number of Processors requested: ', Nproc)
      dir_exist = os.path.isdir(fullDirPath)
   if dir_exist:
      print('Currently directory ' + fullDirPath + ' already exist!')
   else:
      print('Creating directory ' + dirName + ' in' + location)
      os.mkdir(dirName)
      print('Done creating directory ' + fullDirPath)

fname = "cylindrical_rmi_2d_hdf5_chk_"	   # Establish base name and path for the simulation output files
#fieldName = 'mach_number'
fieldName = ['velocity_x', 'velocity_y','sound_speed']
dirName = fieldName[1] + '_map'
location = os.getcwd()
fullDirPath = location + '/' + dirName
path = "/home2/proanoe/Simulations2018/2D/single_mode/run1_Ding_case1/quarter/a_1mm/test6/"

# mproc.cpu_count() <= Nproc
# Initialize MPI Communicator
#comm = MPI.COMM_WORLD
#Nproc = int(comm.Get_size())
#Pid = int(comm.Get_rank())
Pid=0
Nproc=1

    
min_field = 1.0E-03
max_field = 5E-02
initfile = 0
finalfile = 250
nfiles = finalfile - initfile
nSnaps = nfiles
nVars = 3				# number of variables composing the variable
res = 1024				# Get resolution of a squared-cartesian grid
#q = np.zeros([res,res,nvars,nSnaps])	# Initialize variable for correlation

# Divide equal amount of work to each worker (Done by master thread)
start = initfile-initfile
fin = nfiles
local_start = int(start + Pid*(fin-start)/Nproc)
local_end = int(local_start + (fin-start)/Nproc)
local_nfiles = local_end - local_start
if Pid == 0:
	files = pod.getFileName(initfile, finalfile, path, fname)
	average = pod.FlowFavreAverage(nfiles, path, fieldName, files)

for i in range(local_start, local_end):    # complete the file name with the correct index
   print( "Reading file ", files[i], ' with processor ', Pid)
   # Load file
   mylog.setLevel(40)                  # Show no INFO in command prompt
   ds = yt.load(path+files[i])
   #all_data = ds.covering_grid(0, ds.domain_left_edge, [1024,1024,1])	# Extract data in readable form
   all_data = ds.all_data()
   velx = all_data['velocity_x'].value
   vely = all_data['velocity_y'].value
   sndsp= all_data['sound_speed'].value
   if i == 0:
	   q = np.zeros((nVars*(res**2), nfiles))
   ## Start computing the fluctuations
   velx_f = velx-average[0,:]
   vely_f = vely-average[1,:]
   sndsp_f= sndsp-average[2,:]
   q_aux = np.concatenate((velx_f, vely_f, sndsp_f))
   q[:,i] = q_aux 
R = np.dot(np.transpose(q),q)/nSnaps					# Compute Auto-correlation matrix
try:
	eigvects, s, vh = np.linalg.svd(R, full_matrices=True)	# Singular-Value Decomposition
except LinAlgError:
	sys.error('ERROR: SVD did not converged...!')
min_mode = 0
max_mode = 1
PODmodes = pod.getPODmodes(q, eigvects, nSnaps)			# Compute POD modes
aCoeff = pod.getTimeCoeffs(q, PODmodes, nSnaps)			# Compute time coefficients
# Reconstruct with desired modes
qFluc_POD = pod.getPODFlow(aCoeff, PODmodes, min_mode, max_mode)		
# Allocate memory for POD variables
velx_POD = np.zeros((res**2))
vely_POD = np.zeros((res**2))
sndsp_POD= np.zeros((res**2))

vel_min = 0.#np.sqrt(velx.min()**2+vely.min()**2)
vel_max = 4.7e+04#np.sqrt(velx.max()**2+vely.max()**2)

# Reconstruct POD modes for each variable with Reynolds Decomposition
# One-dimensional reconstruction in Morton Z-order curve
for i in range(0, nSnaps):
	ds = yt.load(path+files[i])
	velx_POD[:] = qFluc_POD[0:res**2,i] + average[0,:]
	vely_POD[:] = qFluc_POD[res**2:2*res**2,i] + average[1,:]
	sndsp_POD[:]= qFluc_POD[2*res**2:3*res**2,i] + average[2,:]
	ds.add_field('velx_POD', function=_velx_POD, 
	             sampling_type='cell',
	             display_name="POD velocity x")
	ds.add_field('vely_POD', function=_vely_POD,
	             sampling_type='cell',
	             display_name="POD velocity y")
	ds.add_field('velmag_POD', function=_velmag_POD,
	             sampling_type='cell',
	             display_name="POD velocity")
	#ds.add_field('velocity_fluctuations', function=_velocity_Fluc,
	#             sampling_type='cell',
	#             display_name="POD velocity fluctuations")
	slc = yt.plot_2d(ds, 'velmag_POD')
	slc.set_log("velmag_POD", False)
	slc.set_zlim('velmag_POD', vel_min, vel_max)
	slc.set_cmap("velmag_POD", 'binary')
	slc.save()
	
	slc = yt.SlicePlot(ds, 'z', 'velocity_magnitude')
	slc.set_log('velocity_magnitude', False)
	slc.set_zlim('velocity_magnitude', vel_min, vel_max)
	slc.set_cmap('velocity_magnitude', 'binary')
	slc.save()
print("Success! POD process terminated.")

#level = 0
#dims = ds.domain_dimensions * ds.refine_by**level

#domain = ds.covering_grid(level, 
#                          left_edge=ds.domain_left_edge*3,
#                          dims=dims, fields=['density'])

#f = h5py.File('test.h5', 'w')
#f.create_dataset('/density', data=domain['density'])
#f.close()
