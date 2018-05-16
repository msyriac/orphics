from __future__ import print_function
from orphics import maps,io,cosmology,symcoupling as sc,stats,lensing
from enlib import enmap,bench
import numpy as np
import os,sys



cache = True
hdv = False
deg = 5
px = 1.5
shape,wcs = maps.rect_geometry(width_deg = deg,px_res_arcmin=px)
mc = sc.LensingModeCoupling(shape,wcs)
pols = ['TT',"TE",'EE','EB','TB']


theory = cosmology.default_theory(lpad=20000)
noise_t = 10.0
noise_p = 10.0*np.sqrt(2.)
fwhm = 1.5
kbeam = maps.gauss_beam(fwhm,mc.modlmap)
ells = np.arange(0,3000,1)
lbeam = maps.gauss_beam(fwhm,ells)
ntt = np.nan_to_num((noise_t*np.pi/180./60.)**2./kbeam**2.)
nee = np.nan_to_num((noise_p*np.pi/180./60.)**2./kbeam**2.)
nbb = np.nan_to_num((noise_p*np.pi/180./60.)**2./kbeam**2.)
lntt = np.nan_to_num((noise_t*np.pi/180./60.)**2./lbeam**2.)
lnee = np.nan_to_num((noise_p*np.pi/180./60.)**2./lbeam**2.)
lnbb = np.nan_to_num((noise_p*np.pi/180./60.)**2./lbeam**2.)


ellmin = 20
ellmax = 3000
xmask = maps.mask_kspace(shape,wcs,lmin=ellmin,lmax=ellmax)
ymask = xmask

Als = {}
for pol in pols:
    with bench.show("ALcalc"):
        AL = mc.AL(pol,xmask,ymask,ntt,nee,nbb,theory=theory,hdv=hdv,cache=cache)
    Als[pol] = AL.copy()
bin_edges = np.arange(10,2000,40)

pl = io.Plotter(yscale='log')
pl.add(ells,theory.gCl('kk',ells),lw=3,color='k')

crosses = [('TT','EE'),('TT','TE'),('EE','TE'),('EB','TB')]

for pol1,pol2 in crosses:
    print(pol1,pol2)
    with bench.show("ALcalc"):
        cross = mc.cross(pol1,pol2,theory,xmask,ymask,noise_t=ntt,noise_e=nee,noise_b=nbb,
                      ynoise_t=None,ynoise_e=None,ynoise_b=None,
                      cross_xnoise_t=None,cross_ynoise_t=None,
                      cross_xnoise_e=None,cross_ynoise_e=None,
                      cross_xnoise_b=None,cross_ynoise_b=None,
                      theory_norm=None,hdv=hdv,save_expression="current",validate=True,cache=True)

    Nlalt = np.abs(mc.NL(Als[pol1],Als[pol2],cross))
    cents,nkkalt = stats.bin_in_annuli(Nlalt,mc.modlmap,bin_edges)
    pl.add(cents,nkkalt,marker="o",alpha=0.2,label=pol1 + "x" + pol2)
pl.legend()
pl.done()

zcrosses = [('TT','TB'),('TT','EB'),('EE','EB'),('EE','TB')]

pl = io.Plotter()

for pol1,pol2 in zcrosses:
    print(pol1,pol2)
    with bench.show("ALcalc"):
        cross = mc.cross(pol1,pol2,theory,xmask,ymask,noise_t=ntt,noise_e=nee,noise_b=nbb,
                      ynoise_t=None,ynoise_e=None,ynoise_b=None,
                      cross_xnoise_t=None,cross_ynoise_t=None,
                      cross_xnoise_e=None,cross_ynoise_e=None,
                      cross_xnoise_b=None,cross_ynoise_b=None,
                      theory_norm=None,hdv=hdv,save_expression="current",validate=True,cache=True)

    Nlalt = mc.NL(Als[pol1],Als[pol2],cross)
    cents,nkkalt = stats.bin_in_annuli(Nlalt,mc.modlmap,bin_edges)
    pl.add(cents,nkkalt,marker="o",alpha=0.2,label=pol1 + "x" + pol2)

pl.legend()
pl.done()

print("nffts : ",mc.nfft,mc.nifft)
