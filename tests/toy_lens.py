from __future__ import print_function
from orphics import maps,io,cosmology,lensing,stats
from enlib import enmap,lensing as enlensing,bench
import numpy as np
import os,sys
from alhazen.halos import nfw_kappa
from szar import counts

theory_file_root = "data/Aug6_highAcc_CDM"
cc = counts.ClusterCosmology(skipCls=True)
theory = cosmology.loadTheorySpectraFromCAMB(theory_file_root,unlensedEqualsLensed=False,
                                                    useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=False)
shape, wcs = maps.rect_geometry(width_arcmin=10.,px_res_arcmin=0.5)
lmax = 8000
ells = np.arange(0,lmax,1)
ps = theory.uCl('TT',ells).reshape((1,1,lmax))



modrmap = enmap.modrmap(shape,wcs)
modlmap = enmap.modlmap(shape,wcs)
ksigma = 4.0 * np.pi/180./180.
kamp = 0.1
kappa = kamp*np.exp(-modrmap**2./2./ksigma**2.)
tkmask = maps.mask_kspace(shape,wcs,lmin=100,lmax=6000)
kkmask = maps.mask_kspace(shape,wcs,lmin=100,lmax=7000)
qest = lensing.qest(shape,wcs,theory,kmask=tkmask,kmask_K=kkmask)

phi,_ = lensing.kappa_to_phi(kappa,modlmap,return_fphi=True)
grad_phi = enmap.grad(phi)
        
lens_order = 5

N = 1000

bin_edges = np.arange(0,10.,1.0)
binner = stats.bin2D(modrmap*180.*60/np.pi,bin_edges)
mystats = stats.Stats()
fkappa = maps.filter_map(kappa,kkmask)
cents,yt = binner.bin(fkappa)

for i in range(N):
    unlensed = mg.get_map()

    
    lensed = enlensing.lens_map(unlensed, grad_phi, order=lens_order, mode="spline", border="cyclic", trans=False, deriv=False, h=1e-7)
    recon = qest.kappa_from_map("TT",lensed)
    cents,recon1d = binner.bin(recon)
    mystats.add_to_stack("recon",recon)
    mystats.add_to_stats("recon1d",recon1d)
    mystats.add_to_stats("recon1d_diffper",(recon1d-yt)/yt)
    if (i+1)%100==0: print(i+1)

mystats.get_stacks()
mystats.get_stats()

recon_stack = mystats.stacks['recon']
io.plot_img(kappa,"toy_1_kappa.png")
io.plot_img(unlensed,"toy_2_unlensed.png")
io.plot_img(lensed,"toy_3_lensed.png")
io.plot_img(lensed-unlensed,"toy_4_diff.png")
io.plot_img(recon,"toy_5_recon.png")
io.plot_img(recon_stack,"toy_6_recon_stack.png")
io.plot_img((recon_stack-fkappa)*100./fkappa,"toy_7_recon_diff.png",lim=[-10.,10.])

y = mystats.stats['recon1d']['mean']
yerr = mystats.stats['recon1d']['errmean']

pl = io.Plotter()
pl.add(cents,yt,ls="--")
pl.add_err(cents,y,yerr=yerr,ls="-")
pl.hline()
pl.done("toy_8_recon1d.png")

y = mystats.stats['recon1d_diffper']['mean']
yerr = mystats.stats['recon1d_diffper']['errmean']

pl = io.Plotter()
pl.add_err(cents,y,yerr=yerr,ls="-")
pl.hline()
pl._ax.set_ylim(-0.1,0.02)
pl.done("toy_9_recon1d_diff.png")
