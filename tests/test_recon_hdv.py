from __future__ import print_function
from orphics import maps,io,cosmology,lensing,stats,mpi
from enlib import enmap,lensing as enlensing,bench
import numpy as np
import os,sys
from szar import counts
from scipy.linalg import pinv2
import argparse
import json

# Parse command line
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("Nclusters", type=int,help='Number of simulated clusters.')
parser.add_argument("Amp", type=float,help='Amplitude of mass wrt 1e15.')
parser.add_argument("-a", "--arc",     type=float,  default=100.,help="Stamp width (arcmin).")
parser.add_argument("-p", "--pix",     type=float,  default=0.2,help="Pix width (arcmin).")
parser.add_argument("-b", "--beam",     type=float,  default=1.0,help="Beam (arcmin).")
parser.add_argument("-n", "--noise",     type=float,  default=3.0,help="Noise (uK-arcmin).")
#parser.add_argument("-f", "--flag", action='store_true',help='A flag.')
args = parser.parse_args()


# MPI
comm = mpi.MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()


# Paths

PathConfig = io.load_path_config()
pout_dir = PathConfig.get("paths","plots")+"qest_hdv_"+str(args.noise)+"_"
io.mkdir(pout_dir,comm)


# Theory
theory_file_root = "../alhazen/data/Aug6_highAcc_CDM"
cc = counts.ClusterCosmology(skipCls=True)
theory = cosmology.loadTheorySpectraFromCAMB(theory_file_root,unlensedEqualsLensed=False,
                                                    useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=False)

# Geometry
shape, wcs = maps.rect_geometry(width_arcmin=args.arc,px_res_arcmin=args.pix,pol=False)
modlmap = enmap.modlmap(shape,wcs)
modrmap = enmap.modrmap(shape,wcs)

# Binning
bin_edges = np.arange(0.,20.0,args.pix*2)
binner = stats.bin2D(modrmap*60.*180./np.pi,bin_edges)

# Noise model
noise_uK_rad = args.noise*np.pi/180./60.
normfact = np.sqrt(np.prod(enmap.pixsize(shape,wcs)))
kbeam = maps.gauss_beam(args.beam,modlmap)


# Simulate
lmax = int(modlmap.max()+1)
ells = np.arange(0,lmax,1)
ps = theory.uCl('TT',ells).reshape((1,1,lmax))
ps_noise = np.array([(noise_uK_rad)**2.]*ells.size).reshape((1,1,ells.size))
mg = maps.MapGen(shape,wcs,ps)
ng = maps.MapGen(shape,wcs,ps_noise)
kamp_true = args.Amp
kappa = lensing.nfw_kappa(kamp_true*1e15,modrmap,cc,overdensity=200.,critical=True,atClusterZ=True)
phi,_ = lensing.kappa_to_phi(kappa,modlmap,return_fphi=True)
grad_phi = enmap.grad(phi)
posmap = enmap.posmap(shape,wcs)
pos = posmap + grad_phi
alpha_pix = enmap.sky2pix(shape,wcs,pos, safe=False)
lens_order = 5


if rank==0: print("Starting sims...")
# Stats
Nsims = args.Nclusters
Njobs = Nsims
num_each,each_tasks = mpi.mpi_distribute(Njobs,numcores)
if rank==0: print ("At most ", max(num_each) , " tasks...")
my_tasks = each_tasks[rank]
mstats = stats.Stats(comm)
np.random.seed(rank)


# QE
tellmin = modlmap[modlmap>2].min(); tellmax = 8000; kellmin = tellmin ; kellmax = 8096
tmask = maps.mask_kspace(shape,wcs,lmin=tellmin,lmax=tellmax)
kmask = maps.mask_kspace(shape,wcs,lmin=kellmin,lmax=kellmax)
qest = lensing.qest(shape,wcs,theory,noise2d=kbeam*0.+(noise_uK_rad)**2.,beam2d=kbeam,kmask=tmask,kmask_K=kmask,pol=False,grad_cut=2000,unlensed_equals_lensed=False)

for i,task in enumerate(my_tasks):
    if (i+1)%10==0 and rank==0: print(i+1)

    unlensed = mg.get_map()
    noise_map = ng.get_map()
    lensed = maps.filter_map(enlensing.displace_map(unlensed, alpha_pix, order=lens_order),kbeam)
    stamp = lensed  + noise_map
    if task==0: io.plot_img(stamp,pout_dir+"cmb_noisy.png")

    recon = qest.kappa_from_map("TT",stamp)
    cents, recon1d = binner.bin(recon)

    mstats.add_to_stats("recon1d",recon1d)
    mstats.add_to_stack("recon",recon)

mstats.get_stats()
mstats.get_stacks()

if rank==0:

    stack = mstats.stacks['recon']

    recon1d = mstats.stats['recon1d']['mean']
    recon1d_err = mstats.stats['recon1d']['errmean']
    recon1d_cov = mstats.stats['recon1d']['covmean']

    io.plot_img(stack,pout_dir+"stack.png")

    kappa_true = maps.filter_map(lensing.nfw_kappa(kamp_true*1e15,modrmap,cc,overdensity=200.,critical=True,atClusterZ=True),kmask)
    cents, ktrue1d = binner.bin(kappa_true)

    arcs,ks = np.loadtxt("input/hdv.csv",unpack=True,delimiter=",")
    
    pl = io.Plotter()
    pl.add(cents,ktrue1d,lw=2,color="k")
    pl.add(arcs,ks,lw=2,alpha=0.5)
    pl.add_err(cents,recon1d,recon1d_err,ls="--")
    pl.hline()
    pl.done(pout_dir+"recon1d.png")


    
    pl1 = io.Plotter(xlabel="$A$",ylabel="$\\mathrm{ln}\\mathcal{L}$")
    pl2 = io.Plotter(xlabel="$A$",ylabel="$\\mathcal{L}$")
    

    arcmaxes = [10.0] #[5.0,7.5,10.0,15.0,20.0,30.0]
    for j,arcmax in enumerate(arcmaxes):

        length = cents[cents<=arcmax].size
        cinv = np.linalg.pinv(recon1d_cov[:length,:length])

        
        diff = ktrue1d[cents<=arcmax] 
        chisq = np.dot(np.dot(diff.T,cinv),diff)
        sn = np.sqrt(chisq)
        print("=== ARCMAX ",arcmax," =====")
        print ("S/N null for 1000 : ",sn*np.sqrt(1000./args.Nclusters))
        pred_sigma = kamp_true/sn

        if j==0:
            num_amps = 100
            nsigma = 10.
            kamps = np.linspace(kamp_true-nsigma*pred_sigma,kamp_true+nsigma*pred_sigma,num_amps)
            k1ds = []
            for amp in kamps:
                template = lensing.nfw_kappa(amp*1e15,modrmap,cc,overdensity=200.,critical=True,atClusterZ=True)
                kappa_sim = maps.filter_map(template,kmask)
                cents, k1d = binner.bin(kappa_sim)
                k1ds.append(k1d)


        
        # Fit true for S/N
        lnlikes = []
        for k1d in k1ds:
            
            diff = (k1d-ktrue1d)[cents<=arcmax] 
            lnlike = -0.5*np.dot(np.dot(diff.T,cinv),diff)
            lnlikes.append(lnlike)


        lnlikes = np.array(lnlikes)

        amaxes = kamps[np.isclose(lnlikes,lnlikes.max())]

        pl1.add(kamps,lnlikes)
        p = np.polyfit(kamps,lnlikes,2)
        pl1.add(kamps,p[0]*kamps**2.+p[1]*kamps+p[2],ls="--")
        for amax in amaxes:
            pl1.vline(x=amax,ls="-")

        c,b,a = p
        mean = -b/2./c
        sigma = np.sqrt(-1./2./c)
        print(mean,sigma)
        sn = (kamp_true/sigma)
        print ("S/N fit for 1000 : ",sn*np.sqrt(1000./args.Nclusters))
        pbias = (mean-kamp_true)*100./kamp_true
        #print ("Bias : ",pbias, " %")
        #print ("Bias : ",(mean-kamp_true)/sigma, " sigma")

        like = np.exp(lnlikes)
        like /= like.max()




        nkamps = np.linspace(kamps.min(),kamps.max(),1000)
        pl2.add(nkamps,np.exp(-(nkamps-mean)**2./2./sigma**2.))
        pl2.add(kamps,like)







        # Fit data for bias
        lnlikes = []
        for k1d in k1ds:
            
            diff = (k1d-recon1d)[cents<=arcmax] 
            lnlike = -0.5*np.dot(np.dot(diff.T,cinv),diff)
            lnlikes.append(lnlike)

        lnlikes = np.array(lnlikes)

        amaxes = kamps[np.isclose(lnlikes,lnlikes.max())]
        p = np.polyfit(kamps,lnlikes,2)

        c,b,a = p
        mean = -b/2./c
        sigma = np.sqrt(-1./2./c)
        print(mean,sigma)
        sn = (kamp_true/sigma)
        pbias = (mean-kamp_true)*100./kamp_true
        print ("Bias : ",pbias, " %")
        print ("Bias : ",(mean-kamp_true)/sigma, " sigma")



        



    pl1.vline(x=kamp_true,ls="--")
    pl1.done(pout_dir+"lensed_lnlikes_all.png")
    
    pl2.vline(x=kamp_true,ls="--")
    pl2.done(pout_dir+"lensed_likes.png")
