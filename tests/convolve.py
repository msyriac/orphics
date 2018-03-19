from __future__ import print_function
from orphics import maps,io,cosmology,stats
from enlib import enmap
import numpy as np
import os,sys

"""
What I've learned from this script.

There's no difference between PS calculated on
sims that have been kspace convolved with a gaussian beam
and real-space convolved with a gaussian beam with nsigma=5.

BUT

There's a 2% difference between PS calculated on sims that
have been kspace convolved with a realistic beam and 
real-space convolved with a realistic beam for nsigma=40-200.
For lower nsigma=5, the difference is much larger, up to 40%.

For the realisitic case, real-space convolution seems to induce
pixel offsets. This is very dangerous for lensing simulations
that might cross-correlate with (non-offset) input kappa to
get MC normalization corrections.

Real-space convolved maps also tend to induce non-periodicities
which require a taper for subsequent analysis.

"""


deg = 10.
px = 0.5

shape,wcs,modlmap,cc,mgen = maps.flat_sim(deg,px,lmax=6000,lensed=True,pol=False)
fc = maps.FourierCalc(shape,wcs)

nsims = 10

brealfile = '/gpfs01/astro/workarea/msyriac/data/act/beams/171206/beam_profile_171206_pa1_150GHz_3_310_s15_map.txt'
bfourierfile = '/gpfs01/astro/workarea/msyriac/data/act/beams/171206/beam_tform_171206_pa1_150GHz_3_310_s15_map.txt'

rs,brs = np.loadtxt(brealfile,usecols=[0,1],unpack=True)
rs *= np.pi/180.
ls,bells = np.loadtxt(bfourierfile,usecols=[0,1],unpack=True)


fwhm_test = 1.4
btest = maps.gauss_beam(ls,fwhm_test)

# pl = io.Plotter()
# pl.add(ls,bells)
# pl.add(ls,btest,ls="--")
# pl.done(io.dout_dir+'kbeam.png')


btest = maps.gauss_beam_real(rs,fwhm_test)

# pl = io.Plotter()
# pl.add(rs,brs*rs)
# pl.add(rs,btest*rs,ls="--")
# pl.done(io.dout_dir+'rbeam.png')


modrmap = enmap.modrmap(shape,wcs)
kbeam = maps.interp(ls,bells)(modlmap)
kbeam_test = maps.gauss_beam(modlmap,fwhm_test)


bin_edges = np.arange(40,4000,40)

taper,w2 = maps.get_taper(shape)

st = stats.Stats()

for i in range(nsims):
    print(i)
    imap = mgen.get_map()

    omap1 = maps.convolve_profile(imap.copy(),rs,brs,fwhm_test,nsigma=200.0)
    omap2 = maps.filter_map(imap.copy(),kbeam)
    omap3 = maps.convolve_gaussian(imap.copy(),fwhm=fwhm_test,nsigma=5.0)
    omap4 = maps.filter_map(imap.copy(),kbeam_test)

    if i==0:
        io.plot_img(omap1,io.dout_dir+"drbeam.png",high_res=True)
        io.plot_img(omap2,io.dout_dir+"dfbeam.png",high_res=True)
        io.plot_img(omap3,io.dout_dir+"grbeam.png",high_res=True)
        io.plot_img(omap4,io.dout_dir+"gfbeam.png",high_res=True)

    cents,p0 = maps.binned_power(imap*taper,bin_edges=bin_edges,fc=fc,modlmap=modlmap)
    cents,p1 = maps.binned_power(omap1*taper,bin_edges=bin_edges,fc=fc,modlmap=modlmap)
    cents,p2 = maps.binned_power(omap2*taper,bin_edges=bin_edges,fc=fc,modlmap=modlmap)
    cents,p3 = maps.binned_power(omap3*taper,bin_edges=bin_edges,fc=fc,modlmap=modlmap)
    cents,p4 = maps.binned_power(omap4*taper,bin_edges=bin_edges,fc=fc,modlmap=modlmap)

    st.add_to_stats("p0",p0/w2)
    st.add_to_stats("p1",p1/w2)
    st.add_to_stats("p2",p2/w2)
    st.add_to_stats("p3",p3/w2)
    st.add_to_stats("p4",p4/w2)

st.get_stats()

ells = np.arange(2,6000,1)
lbeam = maps.interp(ls,bells)(ells)
lbeam_test = maps.gauss_beam(ells,fwhm_test)

p0 = st.stats['p0']['mean']
p1 = st.stats['p1']['mean']
p2 = st.stats['p2']['mean']
p3 = st.stats['p3']['mean']
p4 = st.stats['p4']['mean']

pl = io.Plotter(yscale='log')
pl.add(ells,cc.theory.lCl('TT',ells)*ells**2.,color='k',lw=3)
pl.add(ells,cc.theory.lCl('TT',ells)*lbeam**2.*ells**2.,color='k',ls='--',lw=3)
pl.add(cents,p0*cents**2.,label="no beam",lw=2)
pl.add(cents,p2*cents**2.,label="fourier beam",lw=2)
pl.add(cents,p4*cents**2.,label="fourier beam gaussian",lw=2)
pl.add(cents,p3*cents**2.,label="real beam gaussian",lw=2)
pl.add(ells,cc.theory.lCl('TT',ells)*lbeam_test**2.*ells**2.,color='k',ls='-.',lw=3)
pl.add(cents,p1*cents**2.,label="real beam",lw=2)
pl.legend(loc='upper right')
pl.done(io.dout_dir+"convtest.png")

pl = io.Plotter()
pl.add(cents,(p1-p2)/p2)
pl.add(cents,(p3-p4)/p4)
pl.hline()
pl.done(io.dout_dir+"convtestdiff.png")
