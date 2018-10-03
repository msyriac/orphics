import numpy as np
from enlib import bench, enmap
from math import ceil

import multiprocessing

# number of coordinates to transform
N = 96000013

# this should be obtained using multiprocessing.get_cpu_count()
Njobs = 12

th = np.random.uniform(0,90,N)
phi = np.random.uniform(0,90,N)
shape,wcs = enmap.rect_geometry(100.,0.5)
coords = np.array([th,phi])

with bench.show("serial"):
    pix = enmap.pix2sky(shape,wcs,coords)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, l.shape[-1], n):
        yield l[:,i:i + n]
    

size_chunks = int(ceil(coords.shape[1]*1./Njobs))
# print coords.shape
# for i,x in enumerate(chunks(coords,size_chunks)):
#     print i, x.shape
# sys.exit()

def chunked_pix2sky(x):
    pix = enmap.pix2sky(shape,wcs,x)
    return pix

with bench.show("parallel chunked"):
    p = multiprocessing.Pool(Njobs)
    res = np.concatenate(p.map(chunked_pix2sky,  chunks(coords,size_chunks)),axis=1)
    p.terminate()

print pix.shape
print res.shape
assert np.all(np.isclose(pix,res))
