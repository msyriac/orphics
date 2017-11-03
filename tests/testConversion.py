import ctypes

rootPath = "./"

deg2pix = ctypes.CDLL(rootPath + 'bin/deg2healpix.so')
deg2pix.getPixIndexEquatorial.argtypes = (ctypes.c_long, ctypes.c_double, ctypes.c_double)


ra = 30.
dec = -7.
nside = 2048

cnside = ctypes.c_long(nside)

pixE = deg2pix.getPixIndexEquatorial(cnside,ctypes.c_double(ra),ctypes.c_double(dec))
pixG = deg2pix.getPixIndexGalactic(cnside,ctypes.c_double(ra),ctypes.c_double(dec))

print((pixE,pixG))

