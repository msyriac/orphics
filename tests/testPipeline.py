from __future__ import print_function
from enlib import enmap,utils

# hwidth = 10.*60.
# deg = utils.degree
# arcmin =  utils.arcmin
# px = 0.5

# shape_car, wcs_car = enmap.geometry(pos=[[-hwidth*arcmin,-hwidth*arcmin],[hwidth*arcmin,hwidth*arcmin]], res=px*arcmin, proj="car")

# shape_cea, wcs_cea = enmap.geometry(pos=[[-hwidth*arcmin,-hwidth*arcmin],[hwidth*arcmin,hwidth*arcmin]], res=px*arcmin, proj="cea")


# print(shape_car)
# print(wcs_car)

# print(shape_cea)
# print(wcs_cea)

from mpi4py import MPI
from orphics.analysis.pipeline import Pipeline

#p = Pipeline(MPI.COMM_WORLD,num_tasks=23,cores_per_node=8,max_cores_per_node=4)
p = Pipeline(MPI.COMM_WORLD,num_tasks=23)
p.info()
