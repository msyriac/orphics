from __future__ import print_function
from orphics import maps,io,cosmology,lensing,stats
from enlib import enmap,lensing as enlensing,bench
import numpy as np
import os,sys
from szar import counts
from scipy.linalg import pinv2

dimensionless = False
mean_sub = False

theory_file_root = "../alhazen/data/Aug6_highAcc_CDM"
cc = counts.ClusterCosmology(skipCls=True)
theory = cosmology.loadTheorySpectraFromCAMB(theory_file_root,unlensedEqualsLensed=False,
                                                    useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=dimensionless)
shape, wcs = maps.rect_geometry(width_arcmin=10.,px_res_arcmin=0.5,pol=False)
lmax = 8000
ells = np.arange(0,lmax,1)
ps = theory.uCl('TT',ells).reshape((1,1,lmax))
#ps = cosmology.enmap_power_from_orphics_theory(theory,lmax,lensed=False,dimensionless=dimensionless,orphics_dimensionless=dimensionless)
#Ucov = np.load("scov.npy")


Ucov = maps.pixcov(shape,wcs,ps,Nsims=10000,mean_sub=mean_sub,seed=1)
#print(shape,Ucov.shape)
#sys.exit()
# np.save("scov.npy",Ucov)


TCMB = 2.7255e6 if dimensionless else 1.
Ucov /= TCMB**2.

noise_uK_rad = 0.1*np.pi/180./60./TCMB
Ncov = np.diag([(noise_uK_rad)**2.]*Ucov.shape[0])
print(Ucov.shape,Ncov.shape)
# sys.exit()


modrmap = enmap.modrmap(shape,wcs)
modlmap = enmap.modlmap(shape,wcs)
ksigma = 4.0 * np.pi/180./180.
#kamps = np.linspace(0.01,0.8,40)
amp_min = 0.19
amp_max = 0.21
# amp_min = 0.76
# amp_max = 0.84
kamps = np.linspace(amp_min,amp_max,30)

print(kamps)

Ccovs = []
Cinvs = []
logdets = []

#kappa = np.exp(-modrmap**2./2./ksigma**2.)

kappa = lensing.nfw_kappa(1e15,modrmap,cc)

phi,_ = lensing.kappa_to_phi(kappa,modlmap,return_fphi=True)
grad_phi = enmap.grad(phi)
lens_order = 5
posmap = enmap.posmap(shape,wcs)

for k,kamp in enumerate(kamps):
    pos = posmap + kamp*grad_phi
    alpha_pix = enmap.sky2pix(shape,wcs,pos, safe=False)

    Scov = lensing.lens_cov(Ucov,alpha_pix,lens_order=lens_order)
    # io.plot_img(np.nan_to_num((Scov-Ucov)*100./Ucov))
    


    Tcov = Scov + Ncov
    s,logdet = np.linalg.slogdet(Tcov)
    assert s>0
    logdets.append(logdet)
    Cinvs.append( pinv2(Tcov) )
    # Cinvs.append( np.linalg.pinv(Tcov) )
    # print(logdet)
    print(kamp)



lnlikes = []





Nclusters = 1000
slmax = modlmap.max()
ells = np.arange(0,slmax,1)
ps_noise = np.array([(noise_uK_rad)**2.]*ells.size).reshape((1,1,ells.size))
mg = maps.MapGen(shape,wcs,ps)
ng = maps.MapGen(shape,wcs,ps_noise)
kamp_true = 0.2
kappa = kamp_true*lensing.nfw_kappa(1e15,modrmap,cc)
phi,_ = lensing.kappa_to_phi(kappa,modlmap,return_fphi=True)
grad_phi = enmap.grad(phi)
posmap = enmap.posmap(shape,wcs)
pos = posmap + grad_phi
alpha_pix = enmap.sky2pix(shape,wcs,pos, safe=False)
lens_order = 5

aprior = 0.5
aprior_sigma = 0.5
lnprior = (kamps-aprior)**2./aprior_sigma

pl = io.Plotter(xlabel="$A$",ylabel="$\\mathrm{ln}\\mathcal{L}$")
totlikes = 0.
for i in range(Nclusters):
    if (i+1)%10==0: print(i+1)


    
    unlensed = mg.get_map(seed=int(1e9)+i)
    # stamp = unlensed + ng.get_map(seed=int(1e9)+2*i+1)
    noise_map = ng.get_map(seed=int(1e9+1e8)+i)
    stamp = enlensing.displace_map(unlensed, alpha_pix, order=lens_order)  #+ noise_map

    # io.plot_img(unlensed)
    # io.plot_img(stamp)
    
    if mean_sub: stamp -= stamp.mean()


    totlnlikes = []    
    for k,kamp in enumerate(kamps):
        lnlike = maps.get_lnlike(Cinvs[k],stamp) + logdets[k]
        totlnlike = lnlike #+ lnprior[k]
        totlnlikes.append(totlnlike)

    nlnlikes = -0.5*np.array(totlnlikes)
    nlnlikes -= nlnlikes.max()
    totlikes += nlnlikes.copy()
    # print(nlnlikes)

    pl.add(kamps,np.array(nlnlikes),alpha=0.2)
totlikes -= totlikes.max()
totlikes = np.array(totlikes)

amaxes = kamps[np.isclose(totlikes,totlikes.max())]
for amax in amaxes:
    pl.vline(x=amax,ls="-")

# pl._ax.set_ylim(-2e6,1e6)
pl.vline(x=kamp_true,ls="--")
pl.done("lensed_lnlikes.png")

pl = io.Plotter(xlabel="$A$",ylabel="$\\mathrm{ln}\\mathcal{L}$")
pl.add(kamps,totlikes)
p = np.polyfit(kamps,totlikes,2)
pl.add(kamps,p[0]*kamps**2.+p[1]*kamps+p[2],ls="--")
pl.vline(x=kamp_true,ls="--")
for amax in amaxes:
    pl.vline(x=amax,ls="-")
pl.done("lensed_lnlikes_all.png")

c,b,a = p
mean = -b/2./c
sigma = np.sqrt(-1./2./c)
print(mean,sigma)


kamps = np.linspace(amp_min,amp_max,1000)
pl = io.Plotter(xlabel="$A$",ylabel="$\\mathcal{L}$")
pl.add(kamps,np.exp(-(kamps-mean)**2./2./sigma**2.))
pl.vline(x=kamp_true,ls="--")
pl.done("lensed_likes.png")
