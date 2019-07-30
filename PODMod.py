## @package PODMod.py
#  This module contains all necessary functions for 
#  complementing the POD code for simulations obtained
#  by the FLASH code.
#
#  All the members of this module are defined below
#  @author Erik Proano

import numpy as np
from yt.funcs import mylog
from yt.mods import *
from pathlib import Path
import yt

## getFileName return a tuple with the file names
#
#  Based on a base name, this method returns the complene name of files
#  depending on the number of files requested.
#  @param nfiles    The total snapshot count
#  @param path      Path to directory cantaining files
#  @param basename  Common name for the files
def getFileName(initfile, finalfile, path, basename):
	nfiles = finalfile - initfile
	files = [None]*nfiles
	for i in range(initfile, finalfile):
		# Append the correct index to each file
		if i > 9 and i <= 99:
			file = basename + "00" + str(i)
		elif i > 99 and i <= 999:
			file = basename + "0" + str(i)
		elif i > 999 :
		    file = basename + str(i)
		else :
			file = basename + "000" + str(i)
		# Check if file exist or path is correct		
		try:
			my_abs_path = Path(path+file).resolve()
		except FileNotFoundError:
			print("{} file does not exist or path is wrong!\n".format(file))
		else:
			mylog.setLevel(40)
			# print("Reading file {}".format(file))
			files[i-initfile] = file
	return files

## This function performs a Reynolds Decomposition
#  
#  The function extracts the mean flow and the fluc-
#  tuating part of the flow for turbulent flow simu-
#  lations carried out by the FLASH code.
#  @param nfiles    The total snapshot count
#  @param path      Path to directory containing files
#  @param fieldname List with the data fields of interest
#  @param basename  Shared name for the files
def FlowAverage(nfiles, path, fieldname, files):
	print("Starting with averaging process...")
	time = np.zeros(nfiles)				# Initialize time array
	#files = getFileName(nfiles, path, basename)
	j = 0
	for k in range(0,len(fieldname)):
		print("Starting averaging of "+ fieldname[k])
		for i in files:
			print(i)		   
			ds = yt.load(path+i)					# Load the file to the scope
			data = ds.r[fieldname[k]]
			time[j] = ds.current_time
			if j == 0:
				averaged_data = np.zeros(len(data))	# Initialize accumulator
				if k == 0:
					average = np.zeros((len(data),len(fieldname)))
			j += 1
			averaged_data = averaged_data + data
		average[:,k] = averaged_data/nfiles
		j = 0
	print("Done with averaging process...")
	return average
	
	
## FlowFavreAverage provides the Favre averages of flow fields
#
#  This function computes the density-weighted averages for
#  compressible flows.
#  @param nfiles    The total snapshot count
#  @param path      Path to directory containing files
#  @param fieldname List with the data fields of interest
#  @param basename  Shared name for the files
def FlowFavreAverage(nfiles, path, fieldname, files):
	print("Starting with Favre-averaging process...")
	time = np.zeros(nfiles)				# Initialize time array
	dens_flag = "density" in fieldname  # Search if density is defined
	if dens_flag:	# If density is not defined, append it to the list
		dens_id = fieldname.index("density")
	else:
		fieldname.append("density")
		dens_id = fieldname.index("density")
	#files = getFileName(nfiles, path, basename)
	j = 0
	for k in range(0,len(fieldname)):	# Iterate in field name array
		print("Starting averaging of "+ fieldname[k])
		for i in files:					# Iterate in file names array
			print(i)		   
			ds = yt.load(path+i)					# Load the file to the scope
			data = ds.r[fieldname[k]]
			dens_dat = ds.r[fieldname[dens_id]]
			data = data*dens_dat
			time[j] = ds.current_time
			if j == 0:		# If first iteration, initialize array
				averaged_data = np.zeros(len(data))	# Initialize accumulator
				dens_ave_data = np.zeros(len(data))
				if k == 0:
					average = np.zeros((len(data),len(fieldname)))
					dens_ave = np.zeros(len(data))
			j += 1
			dens_ave_data = dens_ave_data + dens_dat
			averaged_data = averaged_data + data
		averaged_data = averaged_data/dens_ave_data
		average[:,k] = averaged_data/nfiles
		j = 0
	print("Done with averaging process...")
	u_tilde, v_tilde, a_tilde = (average[:,0], average[:,1], average[:,2])
	favre_average = np.array((u_tilde, v_tilde, a_tilde))
	print(favre_average.shape)
	return favre_average

## FavreAverage provides the Favre averages of flow fields
#
#  This function computes the density-weighted averages for
#  compressible flows.
#  @param ts	The timeseries array of hdf5 data as read by yt
def FavreAverage(ts):
	dens_ave, ener_ave = 2*[0.]
	velx_ave, vely_ave, velz_ave = 3*[0.]
	n = 0
	for ds in ts:
		currFile = ts.params.data_object.outputs[n]
		print("Reading file " + currFile)
		ad = ds.all_data()
		dens_ave = dens_ave + ad["density"]
		ener_ave = ener_ave + (ad["density"]*ad["total_energy"])
		velx_ave = velx_ave + (ad["density"]*ad["velocity_x"])
		vely_ave = vely_ave + (ad["density"]*ad["velocity_y"])
		velz_ave = velz_ave + (ad["density"]*ad["velocity_z"])
		n += 1
	dens_ave, ener_ave = [dens_ave/n, ener_ave/n]
	velx_ave, vely_ave, velz_ave = [velx_ave/n, vely_ave/n, velz_ave/n]
	velx_ave, vely_ave, velz_ave = [velx_ave/dens_ave, \
                                vely_ave/dens_ave, velz_ave/dens_ave]
	ener_ave = ener_ave/dens_ave
	return (velx_ave, vely_ave, velz_ave, dens_ave, ener_ave)

## Extract POD modes based on eigenvectors
#
#  @param q : Data
#  @param eigenvects : Data eigenvectors
#  @param PODmodes : POD modes of the data
def getPODmodes(q, eigenvects, nSnaps):
	M = len(q)
	PODnorm = np.zeros(nSnaps)
	PODmodes = np.zeros((M, nSnaps))
	for i in range(0, nSnaps):
		PODaux = np.zeros(M)
		for j in range(0, nSnaps):
			aux = eigenvects[j,i]*q[:,j]
			PODaux += aux
			#PODnormAux += aux**2
		PODnorm[i] = np.linalg.norm(PODaux)
		PODmodes[:,i] = PODaux/PODnorm[i] 
	#PODmodes = PODmodes/np.linalg.norm(PODnorm)
	return PODmodes


## Get time coefficients from POD modes
#
#  This function extracts the time coefficients based on
#  the POD modes and the vector of variables.
#  @param q : Vector of variables
#  @param PODmodes : POD modes based on eigenvectors
#  @param aCoeff : Time coefficients for reconstruction
def getTimeCoeffs(q, PODmodes, nSnaps):
	aCoeff = np.zeros((nSnaps, nSnaps))
	for i in range(0, nSnaps):
		aCoeff[:,i] = np.matmul(np.transpose(PODmodes), q[:,i])
	return aCoeff


def getPODFlow(aCoeff, PODmodes, modeInit, modeFin):
	M = len(PODmodes)
	N = len(aCoeff)
	q_POD = np.zeros((M, N))
	for i in range(0, N):
		q_aux = np.zeros(M)
		for j in range(modeInit, modeFin):
			aux = aCoeff[j,i]*PODmodes[:,j]
			q_aux += aux
		q_POD[:,i] = q_aux
	return q_POD
	

# Reconstruct POD modes for each variable with Reynolds Decomposition
# One-dimensional reconstruction in Morton Z-order curve
	for i in range(0, nSnaps):
		velx_POD[:] = qFluc_POD[0:res**2,i] + average[:,0]
	#vely_POD[:,i] = qFluc_POD[res**2:2*res**2,i] + average[:,1]
	#sndsp_POD[:,i]= qFluc_POD[2*res**2:3*res**2,i] + average[:,2]
	return velx_POD


def _velmag(field,data):
	vx = data['gas',"velocity_x"]
	vy = data["gas","velocity_y"]
	nVars = data.get_field_parameter("nVars")
	res = data.get_field_parameter("res")
	nfiles = data.get_field_parameter("nfiles")
	average = data.get_field_parameter("average")
	print(vx.shape)
	print((nVars,res,niles))
	return vx**2 + vy**2


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


## Inner product of a state vector q(rho, u, v, w, T)
##
#  Rho is the density, u, v and w are the x-, y- 
#  and z- components of the velocity field and T is
#  the temperature. All the quantites must be fluct-
#  uating quantities.
#  @param q1 : Array 1
#  @param q2 : Array 2
#  @param a  : weighting factor (see Yang & Fu, 2008)
def compressibleInnerProd(q1, q2, dens_ave, temp_ave, velx_ave, vely_ave, velz_ave, a, res):
	dens1 = q1[0*res**2:1*res**2,:]
	temp1 = q1[1*res**2:2*res**2,:]
	velx1 = q1[2*res**2:3*res**2,:]
	vely1 = q1[3*res**2:4*res**2,:]
	velz1 = q1[4*res**2:5*res**2,:]
	
	dens2 = q2[0*res**2:1*res**2,:]
	temp2 = q2[1*res**2:2*res**2,:]
	velx2 = q2[2*res**2:3*res**2,:]
	vely2 = q2[3*res**2:4*res**2,:]
	velz2 = q2[4*res**2:5*res**2,:]
	print("The shape of each variable tensor is {}".format(dens1.shape))
	vel_ave = np.sqrt(velx_ave**2 + vely_ave**2 + velz_ave**2)
	R = np.zeros((q1.shape[1],q2.shape[1]))
	print("The shape of inner product is {}".format(R.shape))
	for i in range(0, q1.shape[1]):
	#for n in range(0, q1.shape[1]):
		rho1 = yt.YTArray(dens1[:,i], ("g/cm**3"))
		T1 = yt.YTArray(temp1[:,i], ("K"))
		u1 = yt.YTArray(velx1[:,i], ("cm/s"))
		v1 = yt.YTArray(vely1[:,i], ("cm/s"))
		w1 = yt.YTArray(velz1[:,i], ("cm/s"))
		for j in  range(0, q2.shape[1]):
			rho2 = yt.YTArray(dens2[:,j], ("g/cm**3"))
			T2 = yt.YTArray(temp2[:,j], ("K"))
			u2 = yt.YTArray(velx2[:,j], ("cm/s"))
			v2 = yt.YTArray(vely2[:,j], ("cm/s"))
			w2 = yt.YTArray(velz2[:,j], ("cm/s"))
		
			f_dens = np.array((a/2)*(np.dot(rho1,rho2)/np.average(dens_ave**2)))
			f_velx = np.array((a/2)*(np.dot(u1,u2)/np.average(vel_ave**2)))
			f_vely = np.array((a/2)*(np.dot(v1,v2)/np.average(vel_ave**2)))
			f_velz = np.array((a/2)*(np.dot(w1,w2)/np.average(vel_ave**2)))
			f_temp = np.array((1-a)*(np.dot(T1,T2))/np.average(temp_ave**2))
			f = f_dens+f_velx+f_vely+f_velz+f_temp
			R[i,j] = f
	return R


## Inner product of a state vector q(rho, u, v, w, T)
##
#  Rho is the density, u, v and w are the x-, y- 
#  and z- components of the velocity field and T is
#  the temperature. All the quantites must be fluct-
#  uating quantities.
#  @param q1 : Array 1
#  @param q2 : Array 2
#  @param a  : weighting factor (see Yang & Fu, 2008)
def myInnerProd(q1, q2, dens_ave, ener_ave, velx_ave, vely_ave, velz_ave, a, res):
	dens1 = q1[0*res**2:1*res**2,:]
	velx1 = q1[1*res**2:2*res**2,:]
	vely1 = q1[2*res**2:3*res**2,:]
	velz1 = q1[3*res**2:4*res**2,:]
	ener1 = q1[4*res**2:5*res**2,:]
	mfrac1= q1[5*res**2:6*res**2,:]
	
	dens2 = q2[0*res**2:1*res**2,:]
	velx2 = q2[1*res**2:2*res**2,:]
	vely2 = q2[2*res**2:3*res**2,:]
	velz2 = q2[3*res**2:4*res**2,:]
	ener2 = q2[4*res**2:5*res**2,:]
	mfrac2= q2[5*res**2:6*res**2,:]
	print("The shape of each variable tensor is {}".format(dens1.shape))
	vel_ave = np.sqrt(velx_ave**2 + vely_ave**2 + velz_ave**2)
	R = np.zeros((q1.shape[1],q2.shape[1]))
	print("The shape of inner product is {}".format(R.shape))
	for i in range(0, q1.shape[1]):
	#for n in range(0, q1.shape[1]):
		rho1 = yt.YTArray(dens1[:,i], ("g/cm**3"))
		u1 = yt.YTArray(velx1[:,i], ("cm/s"))
		v1 = yt.YTArray(vely1[:,i], ("cm/s"))
		w1 = yt.YTArray(velz1[:,i], ("cm/s"))
		E1 = yt.YTArray(ener1[:,i], ("erg"))
		f1 = yt.YTArray(mfrac1[:,i],(""))
		for j in  range(0, q2.shape[1]):
			rho2 = yt.YTArray(dens2[:,j], ("g/cm**3"))
			u2 = yt.YTArray(velx2[:,j], ("cm/s"))
			v2 = yt.YTArray(vely2[:,j], ("cm/s"))
			w2 = yt.YTArray(velz2[:,j], ("cm/s"))
			E2 = yt.YTArray(ener1[:,i], ("erg"))
			f2 = yt.YTArray(mfrac1[:,i],(""))
		
			f_dens = np.array(np.dot(rho1,rho2))#/np.average(dens_ave**2))
			f_velx = np.array(np.dot(u1,u2)/np.average(vel_ave**2))
			f_vely = np.array(np.dot(v1,v2)/np.average(vel_ave**2))
			f_velz = np.array(np.dot(w1,w2)/np.average(vel_ave**2))
			f_ener = np.array(np.dot(E1,E2)/np.average(ener_ave**2))
			f_mfrac= np.array(np.dot(f1,f2))
			f = f_dens*(1+f_velx+f_vely+f_velz+f_ener+f_mfrac)
			R[i,j] = f
	return R
