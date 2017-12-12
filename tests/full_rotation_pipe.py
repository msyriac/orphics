from __future__ import print_function
from orphics import maps,cosmology,io
from enlib import curvedsky,enmap
import numpy as np
import os,sys

theory_file_root = "../alhazen/data/Aug6_highAcc_CDM"
theory = cosmology.loadTheorySpectraFromCAMB(theory_file_root,unlensedEqualsLensed=False,
                                                    useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=False)
lmax = 2000
ells = np.arange(0,lmax,1)
cls = theory.lCl('TT',ells)
ps = cls.reshape((1,1,ells.size))
fullsky = enmap.read_map("/home/msyriac/data/act/downgraded_template.fits")
#fullsky = enmap.read_map("/home/msyriac/data/act/multitracer_2arc.fits")[0]
fshape, fwcs = fullsky.shape,fullsky.wcs



posmap = enmap.posmap(fshape,fwcs)
DECMIN = posmap[0].min()*180./np.pi
DECMAX = posmap[0].max()*180./np.pi
fullsky[posmap[0]<DECMIN*np.pi/180.] = 0.
fullsky[posmap[0]>DECMAX*np.pi/180.] = 0.
print(posmap.shape)

# io.plot_img(np.flipud(fullsky),"full.png",high_res=True)
print(fshape)

height_dec = 15.
num_decs = int((DECMAX-DECMIN)/height_dec)
print(num_decs)

dec_bounds = np.linspace(DECMIN,DECMAX,num_decs+1)
print(dec_bounds)

ref_0_width_deg = 40.

patches = 0
for k,(dec_0,dec_1) in enumerate(zip(dec_bounds[:-1],dec_bounds[1:])):
    mean_dec = (dec_0+dec_1)/2.
    width = ref_0_width_deg
    num_ras = int(360./width)
    patches += num_ras

    ra_bounds = np.linspace(-180.,180.,num_ras+1)
    for j,(ra_0,ra_1) in enumerate(zip(ra_bounds[:-1],ra_bounds[1:])):
    
        
        test_sky = fullsky.copy()
        test_sky[posmap[0]<dec_0*np.pi/180.] = 0.
        test_sky[posmap[0]>dec_1*np.pi/180.] = 0.

        print(posmap[1].min()*180./np.pi,posmap[1].max()*180./np.pi)
        # sys.exit()
        test_sky[np.logical_not(np.logical_and(posmap[1]<ra_1*np.pi/180.,posmap[1]>ra_0*np.pi/180.))] = 0.

        io.plot_img(np.flipud(test_sky),"test_"+str(k).zfill(2)+"_"+str(j).zfill(2)+".png")
print(patches)
