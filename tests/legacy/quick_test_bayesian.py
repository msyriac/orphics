from __future__ import print_function
from orphics import maps,io,cosmology,lensing,stats
from enlib import enmap,bench,lensing as enlensing,resample
import numpy as np
import os,sys
from szar import counts
import argparse
from scipy.linalg import pinv2

# Parse command line
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("-N", "--Nclusters",     type=int,  default=100,help="Num clusters.")
parser.add_argument("-a", "--arc",     type=float,  default=10.,help="Stamp width (arcmin).")
parser.add_argument("-p", "--pix",     type=float,  default=0.5,help="Pix width (arcmin).")
parser.add_argument("-b", "--beam",     type=float,  default=1.0,help="Beam (arcmin).")
parser.add_argument("-n", "--noise",     type=float,  default=1.0,help="Noise (uK-arcmin).")
parser.add_argument("-f", "--buffer-factor",     type=int,  default=2,help="Buffer factor for stamp.")
#parser.add_argument("-f", "--flag", action='store_true',help='A flag.')
args = parser.parse_args()


# Theory
theory_file_root = "../alhazen/data/Aug6_highAcc_CDM"
cc = counts.ClusterCosmology(skipCls=True)
theory = cosmology.loadTheorySpectraFromCAMB(theory_file_root,unlensedEqualsLensed=False,
                                                    useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=False)



# Geometry

shape, wcs = maps.rect_geometry(width_arcmin=args.arc,px_res_arcmin=args.pix,pol=False)
modlmap = enmap.modlmap(shape,wcs)
modrmap = enmap.modrmap(shape,wcs)
bshape, bwcs = maps.rect_geometry(width_arcmin=args.arc*args.buffer_factor,px_res_arcmin=args.pix,pol=False)
bmodlmap = enmap.modlmap(bshape,bwcs)
bmodrmap = enmap.modrmap(bshape,bwcs)

#gshape, gwcs = maps.rect_geometry(width_arcmin=args.arc,px_res_arcmin=0.1953125,pol=False)
gshape, gwcs = maps.rect_geometry(width_arcmin=100.,px_res_arcmin=args.pix,pol=False)
gshape,gwcs = bshape,bwcs
gmodlmap = enmap.modlmap(gshape,gwcs)
gmodrmap = enmap.modrmap(gshape,gwcs)

print(shape,bshape)


# Noise model
noise_uK_rad = args.noise*np.pi/180./60.
normfact = np.sqrt(np.prod(enmap.pixsize(shape,wcs)))
noise_uK_pixel = noise_uK_rad/normfact
Ncov = np.diag([(noise_uK_pixel)**2.]*np.prod(shape))


# Unlensed signal
    
power2d = theory.uCl('TT',bmodlmap)
bfcov = maps.diagonal_cov(power2d)
sny,snx = shape
ny,nx = bshape
Ucov = maps.pixcov(bshape,bwcs,bfcov)
Ucov = Ucov.reshape(np.prod(bshape),np.prod(bshape))

# Noise model
kbeam = maps.gauss_beam(args.beam,bmodlmap)


# Lens template
lens_order = 5
posmap = enmap.posmap(bshape,bwcs)


# Lens grid
amin = 0.18
amax = 0.22
num_amps = 10
kamps = np.linspace(amin,amax,num_amps)

cinvs = []
logdets = []
for k,kamp in enumerate(kamps):

    kappa_template = lensing.nfw_kappa(kamp*1e15,bmodrmap,cc,overdensity=200.,critical=True,atClusterZ=True)
    phi,_ = lensing.kappa_to_phi(kappa_template,bmodlmap,return_fphi=True)
    grad_phi = enmap.grad(phi)
    pos = posmap + grad_phi
    alpha_pix = enmap.sky2pix(bshape,bwcs,pos, safe=False)

    #if k==0: io.plot_img(kappa_template)

    with bench.show("lensing cov"):
        Scov = lensing.lens_cov(Ucov,alpha_pix,lens_order=lens_order,kbeam=kbeam,bshape=shape)

    Tcov = Scov + Ncov + 5000 # !!!
    with bench.show("covwork"):
        s,logdet = np.linalg.slogdet(Tcov)
        assert s>0
        cinv = pinv2(Tcov).astype(np.float64)
    cinvs.append(cinv)
    logdets.append(logdet)
    print(kamp,logdet)
    # import cPickle as pickle
    # pickle.dump((kamp,logdet,cinv),open("cdump_"+str(k)+".pkl",'wb'))



# Simulate
lmax = int(gmodlmap.max()+1)
ells = np.arange(0,lmax,1)
ps = theory.uCl('TT',ells).reshape((1,1,lmax))
ps_noise = np.array([(noise_uK_rad)**2.]*ells.size).reshape((1,1,ells.size))
mg = maps.MapGen(gshape,gwcs,ps)
ng = maps.MapGen(bshape,bwcs,ps_noise)
kamp_true = 0.2
kappa = lensing.nfw_kappa(kamp_true*1e15,gmodrmap,cc,overdensity=200.,critical=True,atClusterZ=True)
phi,_ = lensing.kappa_to_phi(kappa,gmodlmap,return_fphi=True)
grad_phi = enmap.grad(phi)
posmap = enmap.posmap(gshape,gwcs)
pos = posmap + grad_phi
alpha_pix = enmap.sky2pix(gshape,gwcs,pos, safe=False)
kbeam = maps.gauss_beam(args.beam,gmodlmap)

    
mstats = stats.Stats()

for i in range(args.Nclusters):

    if (i+1)%100==0: print(i+1)
    unlensed = mg.get_map()
    noise_map = ng.get_map()
    lensed = maps.filter_map(enlensing.displace_map(unlensed.copy(), alpha_pix, order=lens_order),kbeam)
    fdownsampled = enmap.enmap(resample.resample_fft(lensed,bshape),bwcs)
    stamp = fdownsampled  + noise_map

    
    #cutout = lensed  + noise_map
    cutout = stamp[int(bshape[0]/2.-shape[0]/2.):int(bshape[0]/2.+shape[0]/2.),int(bshape[0]/2.-shape[0]/2.):int(bshape[0]/2.+shape[0]/2.)]

    # print(cinvs[k].shape,cutout.shape)
    
    totlnlikes = []    
    for k,kamp in enumerate(kamps):
        lnlike = maps.get_lnlike(cinvs[k],cutout) + logdets[k]
        totlnlike = lnlike #+ lnprior[k]
        totlnlikes.append(totlnlike)
    nlnlikes = -0.5*np.array(totlnlikes)
    mstats.add_to_stats("totlikes",nlnlikes)
    
mstats.get_stats()


lnlikes = mstats.vectors["totlikes"].sum(axis=0)
lnlikes -= lnlikes.max()

pl = io.Plotter(xlabel="$A$",ylabel="$\\mathrm{ln}\\mathcal{L}$")
for j in range(mstats.vectors["totlikes"].shape[0]):
    pl.add(kamps,mstats.vectors["totlikes"][j,:]/mstats.vectors["totlikes"][j,:].max())
pl.done(io.dout_dir+"lensed_lnlikes_each_max.png")

pl = io.Plotter(xlabel="$A$",ylabel="$\\mathrm{ln}\\mathcal{L}$")
for j in range(mstats.vectors["totlikes"].shape[0]):
    pl.add(kamps,mstats.vectors["totlikes"][j,:])
pl.done(io.dout_dir+"lensed_lnlikes_each.png")



pl1 = io.Plotter(xlabel="$A$",ylabel="$\\mathrm{ln}\\mathcal{L}$")
pl1.add(kamps,lnlikes,label="bayesian chisquare")
    
amaxes = kamps[np.isclose(lnlikes,lnlikes.max())]
p = np.polyfit(kamps,lnlikes,2)
pl1.add(kamps,p[0]*kamps**2.+p[1]*kamps+p[2],ls="--",label="bayesian chisquare fit")
for amax in amaxes:
    pl1.vline(x=amax,ls="-")


pl1.vline(x=kamp_true,ls="--")
pl1.legend(loc='upper left')
pl1.done(io.dout_dir+"lensed_lnlikes_all.png")

pl2 = io.Plotter(xlabel="$A$",ylabel="$\\mathcal{L}$")

# Bayesian
c,b,a = p
mean = -b/2./c
sigma = np.sqrt(-1./2./c)
print(mean,sigma)
sn = (kamp_true/sigma)
pbias = (mean-kamp_true)*100./kamp_true
print ("BE Bias : ",pbias, " %")
print ("BE Bias : ",(mean-kamp_true)/sigma, " sigma")
print ("S/N for 1000 : ",sn*np.sqrt(1000./args.Nclusters))

like = np.exp(lnlikes)
like /= like.max()
nkamps = np.linspace(kamps.min(),kamps.max(),1000)
pl2.add(nkamps,np.exp(-(nkamps-mean)**2./2./sigma**2.),label="BE likelihood from chisquare fit")
#pl2.add(bkamps,like,label="BE likelihood")


pl2.vline(x=kamp_true,ls="--")
pl2.legend(loc='upper left')
pl2.done(io.dout_dir+"lensed_likes.png")


