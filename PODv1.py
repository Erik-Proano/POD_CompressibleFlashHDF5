###################################################################
##  DESCRIPTION
##    This script computes the POD modes of a Flash simulation with
##    HDF5 data output and performs Galerkin projection of the 
##    compressible 
##
##  AUTHOR
##    Erik S. Proano
##    Embry-Riddle Aeronautical University
##
###################################################################

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import yt
from yt.funcs import mylog
import h5py
import numpy as np
from mpi4py import MPI
import os

def plot_fields(ds, fieldName, path):     
  # Generate and save user-defined field surface plots
  p = yt.plot_2d(ds, fieldName)
  p.set_log(fieldName, False)
  p.set_cmap(fieldName, 'RdBu_r')
  p.save(path)

# Establish base name and path for the simulation output files
fname = "cylindrical_rmi_2d_hdf5_chk_"
#fieldName = 'mach_number'
fieldName = 'density'
dirName = fieldName + '_map'
location = os.getcwd()
fullDirPath = location + '/' + dirName

# Initialize MPI Communicator
comm = MPI.COMM_WORLD
Nproc = int(comm.Get_size())
Pid = int(comm.Get_rank())

#if Pid == 0:
  #print(location)
  #print('Number of Processors requested: ', Nproc)
  #dir_exist = os.path.isdir(fullDirPath)
  #if dir_exist:
    #print('Currently directory ' + fullDirPath + ' already exist!')
  #else:
    #print('Creating directory ' + dirName + ' in' + location)
    #os.mkdir(dirName)
    #print('Done creating directory ' + fullDirPath)
    
min_field = 1.0E-03
max_field = 5E-02
initfile = 0
finalfile = 100
nfiles = finalfile - initfile
nSnaps = nfiles

# Divide equal amount of work to each worker (Done by master thread)
start = initfile
local_start = int(start + Pid*(finalfile-start)/Nproc)
local_end = int(local_start + (finalfile-start)/Nproc)
local_nfiles = local_end - local_start

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
   ad = ds.all_data()
   sl = ds.slice(2,1)
   frb = yt.FixedResolutionBuffer(sl, (0.0,0.0,0.0,4,4,1), (1024,1024))
   my_imag = np.array(frb[fieldName])
   #var1 = np.array(ad['gas', 'velx'])
   #var2 = np.array(ad['gas', 'vely'])
   eigenvals, eigenvecs = np.linalg.eig(my_imag)
