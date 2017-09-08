from orphics.theory.cosmology import Cosmology
import orphics.tools.stats as stats
import numpy as np
import orphics.tools.io as io
from enlib import enmap

ellmin = 2
ellmax = 4000
bin_width = 200

bin_edges = np.arange(ellmin,ellmax,bin_width)

shape, wcs = enmap.get_enmap_patch(width_arcmin=40*60.,px_res_arcmin=0.5)
modlmap = enmap.modlmap(shape,wcs)
binner2d = stats.bin2D(modlmap,bin_edges)
binner1d = stats.bin1D(bin_edges)

cc = Cosmology(lmax=6000,pickling=True)
theory = cc.theory

fine1d_ells = np.arange(ellmin,ellmax,1)

for spec in ['TT','kk']:
    try:
        cl1d = theory.lCl(spec,fine1d_ells)
        cl2d = theory.lCl(spec,modlmap)

    except:
        cl1d = theory.gCl(spec,fine1d_ells)
        cl2d = theory.gCl(spec,modlmap)


        
    cents1d, bin1d = binner1d.binned(fine1d_ells,cl1d)
    cents1d, bin1d_iso = binner1d.binned(fine1d_ells,cl1d*fine1d_ells)
    cents1d, bin1d_iso_norm = binner1d.binned(fine1d_ells,fine1d_ells)
    bin1d_iso /= bin1d_iso_norm
    cents2d, bin2d = binner2d.bin(cl2d)
    
    assert np.all(np.isclose(cents1d,cents2d))

    try:
        interped1d = theory.lCl(spec,cents1d)
    except:
        interped1d = theory.gCl(spec,cents1d)
        
    pdiff = lambda x,y: (x-y)*100./y
    truth = bin2d
    cents = cents2d
    pl = io.Plotter(labelX="$\ell$",labelY="% diff")
    pl.add(cents,pdiff(bin1d,truth),label="1d mean")
    pl.add(cents,pdiff(bin1d_iso,truth),label="1d iso mean")
    pl.add(cents,pdiff(interped1d,truth),label="interpolated")
    pl.legendOn(loc="upper right")
    pl._ax.set_ylim(-5,30)
    pl.done(spec+"_bin.png")
        
