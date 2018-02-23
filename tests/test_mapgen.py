from __future__ import print_function
from orphics import maps,io,cosmology,stats
from enlib import enmap
import numpy as np
import os,sys

shape,wcs = maps.rect_geometry(width_deg=20.,px_res_arcmin=2.0)
cc = cosmology.Cosmology(lmax=6000,pickling=True)

ells = np.arange(2,6000,1)
cltt = cc.theory.lCl('TT',ells)
#cltt = cc.theory.gCl('kk',ells)

ps = cltt.reshape((1,1,ells.size))

mg = maps.MapGen(shape,wcs,ps)
fc = maps.FourierCalc(shape,wcs)

bin_edges = np.arange(40,5000,40)
modlmap = enmap.modlmap(shape,wcs)
binner = stats.bin2D(modlmap,bin_edges)

from scipy.interpolate import interp1d
cltt2d = cc.theory.lCl('TT',modlmap)
#cltt2d = cc.theory.gCl('kk',modlmap)
cents,p1dth = binner.bin(cltt2d)

N = 2000

st = stats.Stats()

for i in range(N):
    if (i+1)%10==0: print(i+1)
    imap = mg.get_map()
    p2d,_,_ = fc.power2d(imap)
    cents,p1d = binner.bin(p2d)

    
    st.add_to_stats("p1diff", (p1d-p1dth)/p1dth)

st.get_stats()

pl = io.Plotter()
pl.add_err(cents,st.stats["p1diff"]['mean'],yerr=st.stats["p1diff"]['errmean'],marker="o",ls="-")
pl.hline()
pl.done()
