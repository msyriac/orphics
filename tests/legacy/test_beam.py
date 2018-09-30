from __future__ import print_function
from orphics import maps,io,cosmology,stats
from enlib import enmap
import numpy as np
import os,sys
from scipy import signal

cc = cosmology.Cosmology(lmax=6000,pickling=True)

deg = 15.
shape,wcs = maps.rect_geometry(width_deg=deg,px_res_arcmin=1.0)
modlmap = enmap.modlmap(shape,wcs)
modrmap = enmap.modrmap(shape,wcs)
lmax = modlmap.max()
ells = np.arange(0,lmax,1)
ps = cc.theory.lCl('TT',ells).reshape((1,1,ells.size))

mgen = maps.MapGen(shape,wcs,ps)




fwhm = 5.
kbeam = maps.gauss_beam(modlmap,fwhm)
imap = mgen.get_map()

bmap2 = maps.convolve_gaussian(imap.copy(),fwhm=fwhm,nsigma=5.0)
print(bmap2.shape)

bmap = maps.filter_map(imap.copy(),kbeam)

taper,w2 = maps.get_taper(shape)

io.plot_img(imap*taper,io.dout_dir+"bmap0.png",high_res=True)
io.plot_img(bmap*taper,io.dout_dir+"bmap1.png",high_res=True)
io.plot_img(bmap2*taper,io.dout_dir+"bmap2.png",high_res=True)

bin_edges = np.arange(200,3000,100)
binner = stats.bin2D(modlmap,bin_edges)
fc = maps.FourierCalc(shape,wcs)
p2d,_,_ = fc.power2d(imap*taper)
cents,p1di = binner.bin(p2d/w2)
p2d,_,_ = fc.power2d(bmap*taper)
cents,p1db = binner.bin(p2d/w2)
p2d,_,_ = fc.power2d(bmap2*taper)
cents,p1db2 = binner.bin(p2d/w2)

pl = io.Plotter(yscale='log')
pl.add(ells,ps[0,0]*ells**2.)
pl.add(ells,ps[0,0]*ells**2.*maps.gauss_beam(ells,fwhm)**2.)
pl.add(cents,p1di*cents**2.)
pl.add(cents,p1db*cents**2.,label="fourier")
pl.add(cents,p1db2*cents**2.,label="real")
pl._ax.set_xlim(2,3000)
pl._ax.set_ylim(1e-12,1e-8)
pl.legend(loc='lower left')
pl.done(io.dout_dir+"beam.png")
