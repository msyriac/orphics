
from enlib import enmap,utils
from alhazen.halos import NFWkappa
from mpi4py import MPI
import numpy as np
import orphics.analysis.pipeline as pipes

wdeg = 100./60.
hwidth = wdeg*60.
deg = utils.degree
arcmin =  utils.arcmin
px = 0.1

shape_car, wcs_car = enmap.geometry(pos=[[-hwidth*arcmin,-hwidth*arcmin],[hwidth*arcmin,hwidth*arcmin]], res=px*arcmin, proj="car")



# === COSMOLOGY ===
cosmologyName = 'LACosmology' # from ini file
iniFile = "../SZ_filter/input/params.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
lmax = 8000
cosmoDict = dictFromSection(Config,cosmologyName)
constDict = dictFromSection(Config,'constants')
cc = ClusterCosmology(cosmoDict,constDict,lmax)
TCMB = 2.7255e6

#=== KAPPA MAP ===
kappaMap,r500 = NFWkappa(cc,massOverh,concentration,zL,thetaMap*180.*60./np.pi,sourceZ,overdensity=overdensity,critical=critical,atClusterZ=atClusterZ)


p = pipes.CMB_Pipeline(MPI.COMM_WORLD, num_tasks = 20,
                       cosmology = None, patch_shape = shape_car, patch_wcs = wcs_car, stride=2)
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
