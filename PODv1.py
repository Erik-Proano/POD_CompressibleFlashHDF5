###################################################################
##  DESCRIPTION
##    This script computes the POD modes of a Flash simulation with
##    HDF5 data output corresponding to a compressible 2D simulation
##    The POD modes, a.k.a. Eigenfunctions, are computed based on 
##    the Singular-Value Decomposition (SVD) provided by numpy API.
##
##  AUTHOR
##    Erik S. Proano
##    Embry-Riddle Aeronautical University
##
###################################################################

from yt.funcs import mylog
import h5py
import numpy as np
from mpi4py import MPI
import multiprocessing as mproc
import os
import PODMod as pod
import Flash_Post as post

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
fieldName = 'density'
dirName = fieldName + '_map'
location = os.getcwd()
fullDirPath = location + '/' + dirName

# mproc.cpu_count() <= Nproc
# Initialize MPI Communicator
comm = MPI.COMM_WORLD
Nproc = int(comm.Get_size())
Pid = int(comm.Get_rank())

    
min_field = 1.0E-03
max_field = 5E-02
initfile = 0
finalfile = 100
nfiles = finalfile - initfile
nSnaps = nfiles
nvars = 3				# number of variables composing the variable
res = 1024				# Get resolution of a squared-cartesian grid
q = np.zeros([res,res,nvars,nSnaps])	# Initialize variable for correlation

# Divide equal amount of work to each worker (Done by master thread)
start = initfile
local_start = int(start + Pid*(finalfile-start)/Nproc)
local_end = int(local_start + (finalfile-start)/Nproc)
local_nfiles = local_end - local_start
if Pid == 0:
   average = pod.ReynoldsDecomp(100, file, fieldname)

for i in range(local_start, local_end):    # complete the file name with the correct index
   if i > 9 and i <= 99:
     file = fname + "00" + str(i)
   elif i > 99 and i <= 999:
     file = fname + "0" + str(i)
   elif i > 999 :
     file = fname + str(i)
   else :
     file = fname + "000" + str(i)

   print( "Reading file ", file, ' with processor ', Pid)
   # Load file
   mylog.setLevel(40)                  # Show no INFO in command prompt
   ds = yt.load(file)
   all_data = ds.covering_grid(0, ds.domain_left_edge, [1024,1024,1])	# Extract data in readable form
   velx = np.array(all_data['velocity_x'])
   vely = np.array(all_data['velocity_y'])
   sndsp= np.array(all_data['sound_speed'])
   q[:,:,0,i] = velx[:,:,0]
   q[:,:,1,i] = vely[:,:,0]
   q[:,:,2,i] = sndsp[:,:,0]
   u, s, vh = np.linalg.svd(var1[:,:,0], full_matrices=True)		# Singular-Value Decomposition
   PODmodes[:,:,i] = u							# POD modes
