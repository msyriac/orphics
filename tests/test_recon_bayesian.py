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
parser.add_argument("GridName", type=str,help='Name of directory to read cinvs from.')
parser.add_argument("Nclusters", type=int,help='Number of simulated clusters.')
parser.add_argument("Amp", type=float,help='Amplitude of mass wrt 1e15.')
#parser.add_argument("-f", "--flag", action='store_true',help='A flag.')
args = parser.parse_args()


# MPI
comm = mpi.MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()


# Paths

PathConfig = io.load_path_config()
GridName = PathConfig.get("paths","output_data")+args.GridName
with open(GridName+"/attribs.json",'r') as f:
    attribs = json.loads(f.read())
arc = attribs['arc'] ; pix = attribs['pix']  ; beam = attribs['beam']  ; noise = attribs['noise']
pout_dir = PathConfig.get("paths","plots")+args.GridName+"/plots_"+io.join_nums((arc,pix,beam,noise))+"_"
io.mkdir(pout_dir,comm)


# Theory
theory_file_root = "../alhazen/data/Aug6_highAcc_CDM"
cc = counts.ClusterCosmology(skipCls=True)
theory = cosmology.loadTheorySpectraFromCAMB(theory_file_root,unlensedEqualsLensed=False,
                                                    useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=False)

# Geometry
shape, wcs = maps.rect_geometry(width_arcmin=arc,px_res_arcmin=pix,pol=False)
modlmap = enmap.modlmap(shape,wcs)
modrmap = enmap.modrmap(shape,wcs)

# Noise model
noise_uK_rad = noise*np.pi/180./60.
kbeam = maps.gauss_beam(beam,modlmap)


# Simulate
lmax = int(modlmap.max()+1)
ells = np.arange(0,lmax,1)
ps = theory.uCl('TT',ells).reshape((1,1,lmax))
ps_noise = np.array([(noise_uK_rad)**2.]*ells.size).reshape((1,1,ells.size))
mg = maps.MapGen(shape,wcs,ps)
ng = maps.MapGen(shape,wcs,ps_noise)
kamp_true = args.Amp
kappa = kamp_true*lensing.nfw_kappa(1e15,modrmap,cc)
phi,_ = lensing.kappa_to_phi(kappa,modlmap,return_fphi=True)
grad_phi = enmap.grad(phi)
posmap = enmap.posmap(shape,wcs)
pos = posmap + grad_phi
alpha_pix = enmap.sky2pix(shape,wcs,pos, safe=False)
lens_order = 5

# Load Cinvs
kamps, logdets = np.loadtxt(GridName+"/amps_logdets.txt",unpack=True)
cinv_file = lambda x: GridName+"/cinv_"+str(x)+".npy"
cinvs = []
for k in range(len(kamps)):
    cinvs.append(np.load(cinv_file(k)))



# Stats
Nsims = args.Nclusters
Njobs = Nsims
num_each,each_tasks = mpi.mpi_distribute(Njobs,numcores)
if rank==0: print ("At most ", max(num_each) , " tasks...")
my_tasks = each_tasks[rank]
mstats = stats.Stats(comm)
np.random.seed(rank)

for i,task in enumerate(my_tasks):
    if (i+1)%10==0 and rank==0: print(i+1)

    unlensed = mg.get_map()
    noise_map = ng.get_map()
    lensed = maps.filter_map(enlensing.displace_map(unlensed, alpha_pix, order=lens_order),kbeam)
    stamp = lensed  + noise_map

    totlnlikes = []    
    for k,kamp in enumerate(kamps):
        lnlike = maps.get_lnlike(cinvs[k],stamp) + logdets[k]
        totlnlike = lnlike #+ lnprior[k]
        totlnlikes.append(totlnlike)

    nlnlikes = -0.5*np.array(totlnlikes)
    mstats.add_to_stats("totlikes",nlnlikes)


mstats.get_stats()

if rank==0:

    totlikes = mstats.vectors["totlikes"].sum(axis=0)
    totlikes -= totlikes.max()

    amaxes = kamps[np.isclose(totlikes,totlikes.max())]

    pl = io.Plotter(xlabel="$A$",ylabel="$\\mathrm{ln}\\mathcal{L}$")
    for nlnlikes in mstats.vectors["totlikes"]:
        pl.add(kamps,np.array(nlnlikes),alpha=0.2)
    for amax in amaxes:
        pl.vline(x=amax,ls="-")
    pl.vline(x=kamp_true,ls="--")
    pl.done(pout_dir+"lensed_lnlikes.png")


    pl = io.Plotter(xlabel="$A$",ylabel="$\\mathrm{ln}\\mathcal{L}$")
    pl.add(kamps,totlikes)
    p = np.polyfit(kamps,totlikes,2)
    pl.add(kamps,p[0]*kamps**2.+p[1]*kamps+p[2],ls="--")
    pl.vline(x=kamp_true,ls="--")
    for amax in amaxes:
        pl.vline(x=amax,ls="-")
    pl.done(pout_dir+"lensed_lnlikes_all.png")

    c,b,a = p
    mean = -b/2./c
    sigma = np.sqrt(-1./2./c)
    print(mean,sigma)


    kamps = np.linspace(kamps.min(),kamps.max(),1000)
    pl = io.Plotter(xlabel="$A$",ylabel="$\\mathcal{L}$")
    pl.add(kamps,np.exp(-(kamps-mean)**2./2./sigma**2.))
    pl.vline(x=kamp_true,ls="--")
    pl.done(pout_dir+"lensed_likes.png")
