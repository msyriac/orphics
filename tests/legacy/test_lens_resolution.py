from __future__ import print_function
from orphics import maps,io,cosmology,lensing
from enlib import enmap, resample, lensing as enlensing
import numpy as np
import os,sys
from szar import counts

mode = "spline"
lens_order = 5
np.random.seed(2)

# Theory
theory_file_root = "../alhazen/data/Aug6_highAcc_CDM"
cc = counts.ClusterCosmology(skipCls=True)
theory = cosmology.loadTheorySpectraFromCAMB(theory_file_root,unlensedEqualsLensed=False,
                                                    useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=False)

lens_func = lambda x: lensing.nfw_kappa(mass,x,cc,zL=0.7,concentration=3.2,overdensity=200.,critical=True,atClusterZ=True)
#sigma = 1.0 * np.pi/180./60.
#lens_func = lambda x: 0.2 * np.exp(-x**2./sigma**2./2.)

rshape,rwcs = maps.rect_geometry(width_arcmin=5.,px_res_arcmin=0.001)
fshape,fwcs = maps.rect_geometry(width_arcmin=20.,px_res_arcmin=0.1)
cshape,cwcs = maps.rect_geometry(width_arcmin=20.,px_res_arcmin=0.5)
rmodrmap = enmap.modrmap(rshape,rwcs)
fmodrmap = enmap.modrmap(fshape,fwcs)
cmodrmap = enmap.modrmap(cshape,cwcs)
rmodlmap = enmap.modlmap(rshape,rwcs)
fmodlmap = enmap.modlmap(fshape,fwcs)
cmodlmap = enmap.modlmap(cshape,cwcs)
print(fshape,cshape)

mass = 2.e14
fkappa = lens_func(fmodrmap)
phi,_ = lensing.kappa_to_phi(fkappa,fmodlmap,return_fphi=True)
grad_phi = enmap.grad(phi)
pos = enmap.posmap(fshape,fwcs) + grad_phi
alpha_pix = enmap.sky2pix(fshape,fwcs,pos, safe=False)


lmax = cmodlmap.max()
ells = np.arange(0,lmax,1)
cls = theory.uCl('TT',ells)
ps = cls.reshape((1,1,ells.size))
mg = maps.MapGen(cshape,cwcs,ps)
unlensed = mg.get_map()


hunlensed = enmap.enmap(resample.resample_fft(unlensed.copy(),fshape),fwcs)
hlensed = enlensing.displace_map(hunlensed, alpha_pix, order=lens_order,mode=mode)
lensed = enmap.enmap(resample.resample_fft(hlensed,cshape),cwcs)

rkappa = lens_func(rmodrmap)
phi,_ = lensing.kappa_to_phi(rkappa,rmodlmap,return_fphi=True)
grad_phi = enmap.grad(phi)
pos = enmap.posmap(rshape,rwcs) + grad_phi
ralpha_pix = enmap.sky2pix(rshape,rwcs,pos, safe=False)

hunlensed = enmap.enmap(resample.resample_fft(unlensed.copy(),rshape),rwcs)
enmap.write_map("unlensed_0.001arc.fits",hunlensed)

hlensed = enlensing.displace_map(hunlensed, ralpha_pix, order=lens_order,mode=mode)
enmap.write_map("lensed_0.001arc.fits",hlensed)
lensed0 = enmap.enmap(resample.resample_fft(hlensed,cshape),cwcs)

sys.exit()


# ckappa = lens_func(cmodrmap) 
# phi,_ = lensing.kappa_to_phi(ckappa,cmodlmap,return_fphi=True)
# grad_phi = enmap.grad(phi)
# pos = enmap.posmap(cshape,cwcs) + grad_phi
# alpha_pix = enmap.sky2pix(cshape,cwcs,pos, safe=False)
# lensed2 = enlensing.displace_map(unlensed, alpha_pix, order=lens_order,mode=mode)

print(lensed.shape)
print(lensed0.shape)

enmap.write_map("lensed_lowres.fits",lensed)
enmap.write_map("lensed_highres.fits",lensed0)
enmap.write_map("unlensed_lowres.fits",unlensed)

io.plot_img(rkappa,"lres_test_00.png")
io.plot_img(lensed0,"lres_test_0.png")
io.plot_img(lensed,"lres_test_1.png")
io.plot_img(lensed0-unlensed,"lres_test_2.png",lim=[-7.,7.])
io.plot_img(lensed-unlensed,"lres_test_3.png",lim=[-7.,7.])
io.plot_img(lensed0-lensed,"lres_test_4.png")
