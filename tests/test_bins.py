from orphics.theory.cosmology import Cosmology
import orphics.tools.stats as stats
import numpy as np
import orphics.tools.io as io
from enlib import enmap

# let's define the bin edges for this test
ellmin = 2
ellmax = 4000
bin_width = 200
bin_edges = np.arange(ellmin,ellmax,bin_width)

# a typical map might be 400sq.deg. with 0.5 arcmin pixels
shape, wcs = enmap.get_enmap_patch(width_arcmin=20*60.,px_res_arcmin=0.5)

# let's get the "mod ell" or |ell| map, which tells us the magnitude of
# the angular wavenumbers in each fourier pixel
modlmap = enmap.modlmap(shape,wcs)

# this let's us create a 2D fourier space binner
binner2d = stats.bin2D(modlmap,bin_edges)

# the 1d binner just needs to know about the bin edges
binner1d = stats.bin1D(bin_edges)

# initialize a cosmology; make sure you have an "output" directory
# for pickling to work
cc = Cosmology(lmax=6000,pickling=True)
theory = cc.theory

# the fine ells we will use
fine1d_ells = np.arange(ellmin,ellmax,1)

# let's test on TT and kk, lCl for TT and gCl for kk
for spec in ['TT','kk']:
    try:
        cl1d = theory.lCl(spec,fine1d_ells) # 1d power spectrum, as obtained from CAMB
        cl2d = theory.lCl(spec,modlmap) # power spectrum on 2d as done with data
    except:
        cl1d = theory.gCl(spec,fine1d_ells)
        cl2d = theory.gCl(spec,modlmap)


    # the mean binning method
    cents1d, bin1d = binner1d.binned(fine1d_ells,cl1d)

    # isotropic mean intended to mimic isotropic 2d spectra
    # this is the most correct way to do it if you don't want to
    # deal with 2d matrices
    cents1d, bin1d_iso = binner1d.binned(fine1d_ells,cl1d*fine1d_ells)
    cents1d, bin1d_iso_norm = binner1d.binned(fine1d_ells,fine1d_ells)
    bin1d_iso /= bin1d_iso_norm

    # the most correctest way to do it if you can work with 2d matrices
    cents2d, bin2d = binner2d.bin(cl2d)
    
    assert np.all(np.isclose(cents1d,cents2d))

    # the not so great way to do it, especially if your spectrum is very
    # red like with TT, or has peaks and troughs like with TT
    try:
        interped1d = theory.lCl(spec,cents1d)
    except:
        interped1d = theory.gCl(spec,cents1d)

    # a percentage difference function
    pdiff = lambda x,y: (x-y)*100./y

    # define 2d binning as the truth
    truth = bin2d
    cents = cents2d

    
    pl = io.Plotter(labelX="$\ell$",labelY="% diff")
    pl.add(cents,pdiff(bin1d,truth),label="1d mean")
    pl.add(cents,pdiff(bin1d_iso,truth),label="1d iso mean")
    pl.add(cents,pdiff(interped1d,truth),label="interpolated")
    pl.legendOn(loc="upper right")
    pl._ax.set_ylim(-5,30)
    pl.done(spec+"_bin.png")
        
