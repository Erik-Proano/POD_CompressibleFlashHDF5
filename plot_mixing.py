#!/usr/share/anaconda35/bin/python
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
from yt.fields.api import ValidateParameter
import yt
import h5py
import numpy as np
from mpi4py import MPI
import multiprocessing as mproc
import os
import PODMod as pod
import Flash_Post as post
import sys
from matplotlib import pyplot as plt

# Generate specific volume field
def _spVol(field, data):
	dens = data['gas', 'density']
	return yt.numpy.power(dens, -1)
  
fname = "cylindrical_rmi_2d_hdf5_chk_"	   # Establish base name and path for the simulation output files
fieldName = ['density', 'specific_volume']
path = os.getcwd()
path = path + '/'

yt.add_field(('gas','specific_volume'), function=_spVol,
                                        units="cm**3/g",
                                        sampling_type="cell",
                                        take_log=False)

r_max = (3.5, "cm")
p_res = 512
r_tar = 2.5
#mproc.cpu_count() <= Nproc
# Initialize MPI Communicator
comm = MPI.COMM_WORLD
Nproc = int(comm.Get_size())
Pid = int(comm.Get_rank())
    
initfile = 0
finalfile = 100
nfiles = finalfile - initfile
nSnaps = nfiles

rad_dens = np.zeros([p_res, nfiles])
rad_spvol= np.zeros([p_res, nfiles])
rad_fld1= np.zeros([p_res, nfiles])
rad_fld2= np.zeros([p_res, nfiles])
b_mix = np.zeros([p_res, nfiles])
d = np.zeros(p_res)
v = np.zeros(p_res)
f1 = np.zeros(p_res)
f2 = np.zeros(p_res)
# Divide equal amount of work to each worker (Done by master thread)
start = initfile
local_start = int(start + Pid*(finalfile-start)/Nproc)
local_end = int(local_start + (finalfile-start)/Nproc)
local_nfiles = local_end - local_start

if Pid == 0:
   files = pod.getFileName(initfile, finalfile, path, fname)
   #average = pod.FlowAverage(nfiles, path, fieldName, files)
   time = np.zeros(nfiles)
   b = np.zeros(nfiles)
else:
   files = None
   time = None
   b = None
p_time = np.zeros(local_nfiles)
p_b = np.zeros(local_nfiles)
comm.Barrier()	# Children wait for Master node to finish
files = comm.bcast(files, root=0)
for i in range(local_start, local_end):    # complete the file name with the correct index
   print( "Reading file ", files[i], ' with processor ', Pid)
   # Load file
   mylog.setLevel(40)                  # Show no yt INFO in command prompt
   ds = yt.load(path+files[i])
   center = ds.domain_left_edge
   sp = ds.sphere(center, r_max)
   rp0 = yt.create_profile(sp, 'radius', fieldName, n_bins=p_res,
                           units = {'radius': 'cm', "density": "kg/m**3", 
                                           "specific_volume": "m**3/kg"},
                           logs = {'radius': False, "density": False, 
                                               'specific_volume': False})
   rp1 = yt.create_profile(sp,"radius",["fld1","fld2"],n_bins=p_res,
                           units = {"radius": "cm", "fld1":"","fld2":""},
                           logs = {"radius":False, "fld1":False, "fld2":False})
   # Transform the profile from a dictionary to a numpy array
   rp0Val1, rp0Val2 = list(rp0.field_data.values())
   rp1Val3, rp1Val4 = list(rp1.field_data.values())
   # Get radial profile values into a 2D array
   rad_dens[:,i] = np.array(rp0["density"]) #d
   rad_spvol[:,i] = np.array(rp0["specific_volume"]) #v
   rad_fld1[:,i] = np.array(rp1["fld1"])#f1
   rad_fld2[:,i] = np.array(rp1["fld2"])#f2

   b_aux = np.mean(rad_dens[:,i])*np.mean(rad_spvol[:,i])
#	dens_f_ave = ad.quantities.weighted_average_quantity("density",'ones') #yt.numpy.average(dens)
#	vfrac_f_ave= ad.quantities.weighted_average_quantity("volume_fraction",'ones') #yt.numpy.average(vfrac)
   p_time[i-local_start] = float(ds.current_time)
   p_b[i-local_start] = b_aux
   b_mix[:,i] = (rad_dens[:,i]*rad_spvol[:,i])-1
   fig = plt.figure()	# Create the figure object
   ax = fig.add_subplot(111)
   ax.plot(rp0.x.value/r_tar, b_mix[:,i],"b-")
   # r^{*} = \frac{R}{R_0}
   ax.set_xlabel(r"$r^{*}$", fontsize=16)
   ax.set_ylabel(r"b", fontsize=16)
   ax.set_ylim(bottom=None, top=1.5, auto=False)
   fig.savefig("b_mixing_t%s" % str(i))
   fig1 = plt.figure()
   ax1 = fig1.add_subplot(111)
   ax1.plot(rp1.x.value/r_tar, rad_fld1[:,i],"b-")
   ax1.plot(rp1.x.value/r_tar, rad_fld2[:,i],"r-")
   ax1.set_xlabel(r"$r^{*}$", fontsize=16)
   ax1.set_ylabel("Mass Fraction", fontsize=16)
   ax1.legend(labels=(r"SF$_{6}$", "Air"), loc="center right")
   fig1.savefig("mfrac_t%s" % str(i))
   plt.close("all")
comm.barrier()						# Wait for all
comm.Gather(p_time, time, root=0)	# Gather time in one array
comm.Gather(p_b, b, root=0)			# Gather enstrophy in one array
MPI.Finalize()						# Finalize communication
# Start ploting enstrophy in time
if Pid == 0:
	radius = np.array(rp0.x.value)
	f = open('b_mixing.dat', 'w')
	f.write('#Radius (cm)\t\t\t\t\t b Parameter at each output file\n')
	np.savetxt('b_mixing.dat', 
                   np.concatenate((np.column_stack(radius).T, 
                                   b_mix),axis=1))
	f.close()
	plt.plot(time*1.E06, b, 'b-')
	plt.xlabel(r'Time ($\mu s$)', fontsize=16)
	plt.ylabel(r'b', fontsize=16)
	#plt.yscale('log')
	plt.grid(True)
	plt.savefig('b_mixing.png')
