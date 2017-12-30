from __future__ import print_function
from orphics.stats import bin2D, Stats
from enlib import enmap, bench, curvedsky, lensing, powspec, mapsim
from orphics import cosmology, io, maps
import numpy as np
import sys
from orphics.mpi import MPI,mpi_distribute
import healpy as hp
import argparse

# Parse command line
parser = argparse.ArgumentParser(description='Make full sky sims')
parser.add_argument("path", type=str,help='Output path prefix')
parser.add_argument("theory", type=str,help='Theory CAMB Cl lensPotential file name')
parser.add_argument("-N", "--nsim",     type=int,  default=1,help="A description")
parser.add_argument("-r", "--res",     type=float,  default=1.0,help="Pixel size in arcminutes")
parser.add_argument("-B", "--bits",     type=int, default=32)
parser.add_argument("-s", "--seed",     type=int,  default=None)
parser.add_argument("-l", "--lmax",     type=int,  default=8000)
parser.add_argument("-m", "--maplmax",  type=int,  default=None)
parser.add_argument("--ncomp",          type=int,  default=3)

#parser.add_argument("-f", "--flag", action='store_true',help='A flag.')
args = parser.parse_args()

dtype= {32:np.float32, 64:np.float64}[args.bits]
seed = args.seed if args.seed is not None else np.random.randint(0,100000000)
lmax = args.lmax or None
maplmax = args.maplmax if args.maplmax is not None else lmax


Nsims = args.nsim

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    
Ntot = Nsims
num_each,each_tasks = mpi_distribute(Ntot,numcores)
mpibox = Stats(comm,tag_start=333)
if rank==0: print("At most ", max(num_each) , " tasks...")
my_tasks = each_tasks[rank]



ps = powspec.read_camb_full_lens(args.theory+"_lenspotentialCls.dat")
shape, wcs = enmap.fullsky_geometry(res=args.res*np.pi/180./60., proj="car")
shape = (args.ncomp,)+shape


def save(suffix,imap,index):
    enmap.write_fits(args.path+"_"+suffix+"_"+str(index).zfill(int(np.log10(args.nsim))+1)+".fits",imap)

for k,index in enumerate(my_tasks):

    if rank==0: print("Rank 0 doing task ", k, " / ", len(my_tasks), "...")

    with bench.show("lensing"):
        lensed,kappa,unlensed = lensing.rand_map(shape, wcs, ps, lmax=lmax, maplmax=maplmax, seed=(seed,index), verbose=True if rank==0 else False, dtype=dtype,output="lku")

    save("lensed",lensed,index)
    save("unlensed",unlensed,index)
    save("kappa",kappa,index)
    
    l_alm = curvedsky.map2alm(lensed,lmax=lmax)
    u_alm = curvedsky.map2alm(unlensed,lmax=lmax)
    k_alm = curvedsky.map2alm(kappa,lmax=lmax)

    del lensed
    del unlensed
    del kappa

    
    lcls = hp.alm2cl(l_alm.astype(np.complex128))
    ucls = hp.alm2cl(u_alm.astype(np.complex128))
    kcls = hp.alm2cl(k_alm.astype(np.complex128))

    del l_alm
    del u_alm
    del k_alm
    
    mpibox.add_to_stack("lcls",lcls)
    mpibox.add_to_stack("ucls",ucls)
    mpibox.add_to_stack("kcls",kcls)


mpibox.get_stacks()

if rank==0:

    theory_file_root = args.theory
    theory = cosmology.loadTheorySpectraFromCAMB(theory_file_root,unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=False)

    
    ucls = mpibox.stacks['ucls']
    lcls = mpibox.stacks['lcls']
    kcls = mpibox.stacks['kcls']
    
    lshape = lcls.shape
    ushape = ucls.shape
    kshape = kcls.shape
    
    dlells = np.arange(lshape[1])
    tltt = theory.lCl('TT',dlells)
    tlte = theory.lCl('TE',dlells)
    tlee = theory.lCl('EE',dlells)
    tlbb = theory.lCl('BB',dlells)
    
    
    duells = np.arange(ushape[1])
    kells = np.arange(kshape[0])
    kk_val = theory.gCl('kk',kells)
    tkk = ps[0,0,:duells.size] *(duells*(duells+1.))**2./4.
    tute = ps[1,2,:duells.size]
    tuee = ps[2,2,:duells.size]
    tubb = ps[3,3,:duells.size]
    tutt = ps[1,1,:duells.size]
    
    # unlensed TT/EE
    pl = io.Plotter(yscale='log',xscale='log',xlabel="$\\ell$",ylabel="$\\ell^2 C_{\\ell}$")
    pl.add(duells,(tutt*duells**2.),lw=2,color='k')
    pl.add(duells,(tuee*duells**2.),lw=2,color='k')
    pl.add(duells,(ucls[0]*duells**2.))
    pl.add(duells,(ucls[1]*duells**2.))
    pl._ax.set_xlim(2,6000)
    pl._ax.set_ylim(1e-6,1e6)
    #pl.done(args.path+"_tt_ee.png")
    pl.done(io.dout_dir+"_unlensed_tt_ee.png")

    # unlensed TE
    pl = io.Plotter(xlabel="$\\ell$",ylabel="$\\ell^2 C_{\\ell}$")
    pl.add(duells,(tute*duells**2.),lw=2,color='k')
    pl.add(duells,(ucls[3]*duells**2.))
    pl._ax.set_xlim(2,6000)
    pl.hline()
    #pl.done(args.path+"_tt_ee.png")
    pl.done(io.dout_dir+"_unlensed_te.png")

    # unlensed BB / null
    pl = io.Plotter(xlabel="$\\ell$",ylabel="$C_{\\ell}$")
    pl.add(duells,(tubb),lw=2,color='k')
    pl.add(duells,(ucls[2]))
    pl.add(duells,(ucls[4]))
    pl.add(duells,(ucls[5]))
    pl._ax.set_xlim(2,6000)
    minval = np.min(np.stack((ucls[2],ucls[4],ucls[5],tubb)))
    maxval = np.max(np.stack((ucls[2],ucls[4],ucls[5],tubb)))
    pl._ax.set_ylim(minval,maxval)
    pl.hline()
    #pl.done(args.path+"_tt_ee.png")
    pl.done(io.dout_dir+"_unlensed_bb_nulls.png")

    # kk
    pl = io.Plotter(yscale='log',xscale='log',xlabel="$L$",ylabel="$C_L$")
    pl.add(duells,(tkk),lw=2,color='k')
    pl.add(kells,(kk_val),lw=2,ls="--",color='k')
    pl.add(kells,(kcls))
    pl._ax.set_xlim(2,7000)
    #pl.done(args.path+"_tt_ee.png")
    pl._ax.set_ylim(1e-11,1e-6)
    pl.done(io.dout_dir+"_kk.png")


    # lensed TT/EE/BB
    pl = io.Plotter(yscale='log',xscale='log',xlabel="$\\ell$",ylabel="$\\ell^2 C_{\\ell}$")
    pl.add(dlells,(tltt*dlells**2.),lw=2,color='k')
    pl.add(dlells,(tlee*dlells**2.),lw=2,color='k')
    pl.add(dlells,(lcls[0]*dlells**2.))
    pl.add(dlells,(lcls[1]*dlells**2.))
    pl.add(dlells,(tlbb*dlells**2.),lw=2,color='k')
    pl.add(dlells,(lcls[2]*dlells**2.))
    pl._ax.set_xlim(2,6000)
    #pl.done(args.path+"_tt_ee.png")
    pl.done(io.dout_dir+"_lensed_tt_ee.png")

    # lensed TE
    pl = io.Plotter(xlabel="$\\ell$",ylabel="$\\ell^2 C_{\\ell}$")
    pl.add(dlells,(tlte*dlells**2.),lw=2,color='k')
    pl.add(dlells,(lcls[3]*dlells**2.))
    pl.hline()
    pl._ax.set_xlim(2,6000)
    #pl.done(args.path+"_tt_ee.png")
    pl.done(io.dout_dir+"_lensed_te.png")

    # lensed nulls
    pl = io.Plotter(xlabel="$\\ell$",ylabel="$C_{\\ell}$")
    pl.add(dlells,(lcls[4]))
    pl.add(dlells,(lcls[5]))
    pl._ax.set_xlim(2,6000)
    minval = np.min(np.stack((lcls[4],lcls[5])))
    maxval = np.max(np.stack((lcls[4],lcls[5])))
    pl._ax.set_ylim(minval,maxval)
    pl.hline()
    #pl.done(args.path+"_tt_ee.png")
    pl.done(io.dout_dir+"_lensed_nulls.png")


    # diffs
    pl = io.Plotter(xlabel="$\\ell$",ylabel="$\\frac{\Delta C_{\\ell}}{C_{\\ell}}$")
    pl.add(duells,(ucls[0]-tutt)/tutt,alpha=0.3,ls="--",color="C0")
    pl.add(duells,(ucls[1]-tuee)/tuee,alpha=0.3,ls="--",color="C1")
    pl.add(duells,(ucls[2]-tubb)/tubb,alpha=0.3,ls="--",color="C2")
    pl.add(dlells,(lcls[0]-tltt)/tltt,alpha=0.8,color="C0",label="tt")
    pl.add(dlells,(lcls[1]-tlee)/tlee,alpha=0.8,color="C1",label="ee")
    pl.add(dlells,(lcls[2]-tlbb)/tlbb,alpha=0.8,color="C2",label="bb")
    pl.add(kells,(kcls-kk_val)/kk_val,alpha=0.8,color="C3",label="kk")

    
    pl._ax.set_xlim(1,7000)
    pl._ax.set_ylim(-0.02,0.02)
    pl.hline()
    pl.legend(loc='lower right')
    #pl.done(args.path+"_tt_ee.png")
    pl.done(io.dout_dir+"_diffs.png")



    pl = io.Plotter(xlabel="$\\ell$",ylabel="$\Delta C_{\\ell}$")
    pl.add(duells,(ucls[3]-tute),alpha=0.3,ls="--",color="C3")
    pl.add(dlells,(lcls[3]-tlte),alpha=0.8,color="C3",label="te")
    pl._ax.set_xlim(1,7000)
    pl._ax.set_ylim(np.min(tlte),np.max(tlte))
    pl.hline()
    pl.legend(loc='lower right')
    #pl.done(args.path+"_tt_ee.png")
    pl.done(io.dout_dir+"_diffs_te.png")

