from orphics import maps, cosmology,io
from enlib import enmap, lensing
import numpy as np

theory_file_root = "../alhazen/data/Aug6_highAcc_CDM"
theory = cosmology.loadTheorySpectraFromCAMB(theory_file_root,unlensedEqualsLensed=False,
                                                    useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=False)

shape,wcs = maps.rect_geometry(width_deg=10.,px_res_arcmin=0.5)

lmax = 4000
ells = np.arange(0,lmax,1)

cltt = theory.uCl('TT',ells)
pstt = cltt.reshape((1,1,ells.size))
cgen = maps.MapGen(shape,wcs,pstt)

imap = cgen.get_map()
clpp = np.nan_to_num(theory.gCl('kk',ells)*4./ells**4.)



psphi = clpp.reshape((1,1,ells.size))
phigen = maps.MapGen(shape,wcs,psphi)
phi = phigen.get_map()
grad_phi = enmap.grad(phi)
omap = lensing.lens_map(imap, grad_phi, order=3, mode="spline", border="cyclic", trans=False, deriv=False, h=1e-7)

io.plot_img(imap,"unlensed.png",high_res=True)
io.plot_img(omap,"lensed.png",high_res=True)
