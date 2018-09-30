import orphics.tools.io as io
import numpy as np

map_root = "/home/msyriac/data/"

file_name = map_root+"shear.fits"

ras = io.read_ignore_error(map_root+"shear_ras.txt")
decs = io.read_ignore_error(map_root+"shear_decs.txt")

print((ras.size))
arr = np.vstack((ras,decs)).T
print((arr.shape))
col_names = ["RA","DEC"]
io.list_to_fits_table(arr,col_names,file_name)
