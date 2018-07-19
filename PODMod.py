import numpy as np
from yt.funcs import mylog
from pathlib import Path
import yt

## This function performs a Reynolds Decomposition
#  
#  The function extracts the mean flow and the fluc-
#  tuating part of the flow for turbulent flow simu-
#  lations carried out by the FLASH code.
#  @param nfiles    The total snapshot count
#  @param path      Path to directory cantaining files
#  @param basename  Common name for the files
#  @param fieldname Name of the data field of interest
def ReynoldsDecomp(nfiles, path, basename, fieldname):
   time = np.zeros(nfiles)				# Initialize time array
   for i in range(nfiles):
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
         print("Reading file {}".format(file))
         ds = yt.load(path+file)			# Load the file to the scope
         data = ds.r[fieldname]
         time[i] = ds.current_time
         if i == 0:
            averaged_data = np.zeros(len(data))	# Initialize accumulator
         averaged_data = averaged_data + data
   average = averaged_data/nfiles
   fluctuations = data - average
   return (average, fluctuations)
