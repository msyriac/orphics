from __future__ import print_function
from orphics import maps,cosmology,io,stats
from enlib import curvedsky,enmap
from scipy.interpolate import interp1d
import numpy as np
import os,sys

theory_file_root = "../alhazen/data/Aug6_highAcc_CDM"
theory = cosmology.loadTheorySpectraFromCAMB(theory_file_root,unlensedEqualsLensed=False,
                                                    useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=False)


ls,nls = np.loadtxt("nlkk.dat",usecols=[0,1],unpack=True)
clkk = theory.gCl('kk',ls)
ellrange = np.arange(0,6000,1)
totcls = interp1d(ls,clkk + nls,bounds_error=False,fill_value="extrapolate")(ellrange)
ps = totcls.reshape((1,1,ellrange.size))
bshape,bwcs = maps.rect_geometry(width_deg = 80.,px_res_arcmin=2.0,height_deg=15.)
tap_per = 1./40.*100.
pad_per = 1./40.*100.

mg = maps.MapGen(bshape,bwcs,ps)
fc = maps.FourierCalc(bshape,bwcs)

taper,w2 = maps.get_taper(bshape,taper_percent = tap_per,pad_percent = pad_per,weight=None)

bmap = mg.get_map()

#io.plot_img(bmap*taper,"map.png",high_res=True)

# bin_edges = np.arange(100,5000,100)
# binner = stats.bin2D(enmap.modlmap(bshape,bwcs),bin_edges)
# cents, bp1d = binner.bin(fc.power2d(bmap*taper)[0]/w2)

# pl = io.Plotter(yscale='log')
# pl.add(ellrange,totcls)
# pl.add(cents,bp1d)
# pl.done("cls.png")

bbox = bmap.box()
pad_rad = 4.*np.pi/180.
bbox0 = bbox.copy()
center = (bbox[1,1]+bbox[0,1])/2.
bbox0[1,1] = center+pad_rad
bbox1 = bbox.copy()
bbox1[0,1] = center-pad_rad

print(bbox*180./np.pi)
print(bbox0*180./np.pi)
print(bbox1*180./np.pi)

bleft = bmap.submap(bbox0)
bright = bmap.submap(bbox1)

print(bleft.shape)
print(bright.shape)

io.plot_img(bmap)
io.plot_img(bleft)
io.plot_img(bright)
