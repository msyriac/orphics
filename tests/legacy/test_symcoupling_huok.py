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
pol = "TE"

# for t in mc.integrands['test']:
#     print(t['l1'])
#     print(t['l2'])
#     print(t['other'])
#     print("----")
# print(len(mc.integrands['test']))

theory = cosmology.default_theory(lpad=20000)
noise_t = 27.0
noise_p = 40.0*np.sqrt(2.)
fwhm = 7.0
# noise_t = 10.0
# noise_p = 14.0*np.sqrt(2.)
# fwhm = 2.0
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

with bench.show("ALcalc"):
    AL = mc.AL(pol,xmask,ymask,ntt,nee,nbb,theory=theory,hdv=hdv,cache=cache)
val = mc.NL_from_AL(AL)

bin_edges = np.arange(10,2000,40)
cents,nkk = stats.bin_in_annuli(val,mc.modlmap,bin_edges)

ls,hunls = np.loadtxt("../alhazen/data/hu_"+pol.lower()+".csv",delimiter=',',unpack=True)
pl = io.Plotter(yscale='log')
pl.add(ells,theory.gCl('kk',ells),lw=3,color='k')
pl.add(cents,nkk,ls="--")
pl.add(ls,hunls*2.*np.pi/4.,ls="-.")

oest = ['TE','ET'] if pol=='TE' else [pol]
ls,nlkks,theory,qest = lensing.lensing_noise(ells,lntt,lnee,lnbb,
                  ellmin,ellmin,ellmin,
                  ellmax,ellmax,ellmax,
                  bin_edges,
                  theory=theory,
                  estimators = oest,
                  unlensed_equals_lensed=False,
                  width_deg=10.,px_res_arcmin=1.0)
    
pl.add(ls,nlkks['mv'],ls="-")

with bench.show("ALcalc"):
    cross = mc.cross(pol,pol,theory,xmask,ymask,noise_t=ntt,noise_e=nee,noise_b=nbb,
                  ynoise_t=None,ynoise_e=None,ynoise_b=None,
                  cross_xnoise_t=None,cross_ynoise_t=None,
                  cross_xnoise_e=None,cross_ynoise_e=None,
                  cross_xnoise_b=None,cross_ynoise_b=None,
                  theory_norm=None,hdv=hdv,save_expression="current",validate=True,cache=True)
    # cross = mc.cross(pol,pol,theory,xmask,ymask,noise_t=ntt,noise_e=nee,noise_b=nbb,
    #               ynoise_t=None,ynoise_e=None,ynoise_b=None,
    #               cross_xnoise_t=0,cross_ynoise_t=0,
    #               cross_xnoise_e=0,cross_ynoise_e=0,
    #               cross_xnoise_b=0,cross_ynoise_b=0,
    #               theory_norm=None,hdv=hdv,save_expression="current",validate=True,cache=True)

Nlalt = mc.NL(AL,AL,cross)
cents,nkkalt = stats.bin_in_annuli(Nlalt,mc.modlmap,bin_edges)
pl.add(cents,nkkalt,marker="o",alpha=0.2)

pl.done()


print("nffts : ",mc.nfft,mc.nifft)
