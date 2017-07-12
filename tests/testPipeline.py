#from __future__ import print_function
from enlib import enmap,utils
from alhazen.halos import NFWkappa

wdeg = 100./60.
hwidth = wdeg*60.
deg = utils.degree
arcmin =  utils.arcmin
px = 0.1

shape_car, wcs_car = enmap.geometry(pos=[[-hwidth*arcmin,-hwidth*arcmin],[hwidth*arcmin,hwidth*arcmin]], res=px*arcmin, proj="car")

# shape_cea, wcs_cea = enmap.geometry(pos=[[-hwidth*arcmin,-hwidth*arcmin],[hwidth*arcmin,hwidth*arcmin]], res=px*arcmin, proj="cea")


# print(shape_car)
# print(wcs_car)

# print(shape_cea)
# print(wcs_cea)

from mpi4py import MPI
import numpy as np
import orphics.analysis.pipeline as pipes

p = pipes.CMB_Pipeline(MPI.COMM_WORLD, num_tasks = 20,
                       cosmology = None, patch_shape = shape_car, patch_wcs = wcs_car,
                       cores_per_node = None, max_cores_per_node = None)
p.loop()

# if p.mrank==0:
#     a = np.random.uniform(size=(2,2))
# else:
#     a = np.empty(shape=(2,2),dtype=np.float64)

# p.distribute(a,tag=1)
# print a

# while True:
#     pass
#p = pipes.CMB_Pipeline(MPI.COMM_WORLD,num_tasks=23,cosmology=None)

#p.info()
