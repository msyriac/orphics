from __future__ import print_function
from orphics import maps,io,cosmology,mpi,stats,lensing
from enlib import enmap,lensing as enlensing
import numpy as np
import os,sys
import argparse

"""

Loads cutouts from full-sky lensed simulations.
Performs QE lensing reconstruction for both temperature
and polarization estimators.
Cross-correlates with input convergence to verify reconstruction.
(Does not verify auto-spectra)

Full-sky sims work to <5% for all estimators if iau=True. Pol estimators
esp. TB are worse, possibly due to E->B leakage.

Flat-sky sims do not get pol right for 2-pt or for lensing, no matter
if iau is False or True.

"""


# Parse command line
parser = argparse.ArgumentParser(description='Verify all estimators.')
parser.add_argument("SimRoot", type=str,help='Path to sims + root name. e.g. /home/msyriac/sims/run1')
parser.add_argument("TheoryRoot", type=str,help='Path to CAMB theory + root name. e.g. /home/msyriac/sims/cosmo')
parser.add_argument("-N", "--nsims",     type=int,  default=10,help="Number of sims to use.")
parser.add_argument("-y", "--wy",     type=float,  default=15.,help="Height of patch in degrees.")
parser.add_argument("-x", "--wx",     type=float,  default=40.,help="Width of patch in degrees.")
parser.add_argument("--beam",     type=float,  default=1.5,help="Beam FWHM in arcmin.")
parser.add_argument("--noise",     type=float,  default=1.5,help="T noise in muK-arcmin.")
parser.add_argument("--taper-width",     type=float,  default=1.5,help="Taper width in degrees.")
parser.add_argument("--pad-width",     type=float,  default=0.5,help="Taper width in degrees.")
parser.add_argument("--tellmin",     type=int,  default=500,help="Temperature ellmin.")
parser.add_argument("--tellmax",     type=int,  default=3000,help="Temperature ellmax.")
parser.add_argument("--pellmin",     type=int,  default=500,help="Polarization ellmin.")
parser.add_argument("--pellmax",     type=int,  default=5000,help="Polarization ellmax.")
parser.add_argument("--kellmin",     type=int,  default=40,help="Convergence ellmin.")
parser.add_argument("--kellmax",     type=int,  default=3000,help="Convergence ellmax.")
parser.add_argument("--dell",     type=int,  default=100,help="Spectra bin-width.")
parser.add_argument("--estimators",     type=str,  default="TT,TE,ET,EB,EE,TB",help="List of polcombs.")
parser.add_argument("--debug", action='store_true',help='Debug with plots.')
parser.add_argument("--iau", action='store_true',help='Use IAU pol convention.')
parser.add_argument("--flat", action='store_true',help='Do flat sky periodic.')
parser.add_argument("-f","--flat-force", action='store_true',help='Force flat sky remake.')
parser.add_argument("--flat-res",     type=float,  default=1.0,help="Resolution in arcmin if flat sky sims.")
parser.add_argument("--flat-taper", action='store_true',help='Taper periodic flat-sky sims.')
#parser.add_argument("-p", "--pad", action='store_true',help='Big if true.')
args = parser.parse_args()
polcombs = args.estimators.split(',')
pol = False if polcombs==['TT'] else True
io.dout_dir += "qev_"
if args.flat: io.dout_dir += "flat_"
if args.iau: io.dout_dir += "iau_"

# MPI
comm = mpi.MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()
Nsims = args.nsims
Njobs = Nsims
num_each,each_tasks = mpi.mpi_distribute(Njobs,numcores)
if rank==0: print ("At most ", max(num_each) , " tasks...")
my_tasks = each_tasks[rank]

# Theory
theory = cosmology.loadTheorySpectraFromCAMB(args.TheoryRoot,
                                             unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=False)

# Geometry
fshape,fwcs = enmap.fullsky_geometry(res=1.0*np.pi/180./60.)
box = np.array([[-args.wy/2.,-args.wx/2.],[args.wy/2.,args.wx/2.]])*np.pi/180.

# Stats
st = stats.Stats(comm)

def init_geometry(ishape,iwcs):
    modlmap = enmap.modlmap(ishape,iwcs)
    bin_edges = np.arange(args.kellmin,args.kellmax,args.dell)
    binner = stats.bin2D(modlmap,bin_edges)
    kbeam = maps.gauss_beam(modlmap,args.beam)
    lmax = modlmap.max()
    ells = np.arange(2,lmax,1)
    wnoise_TT = ells*0.+(args.noise*(np.pi/180./60.))**2.
    wnoise_PP = 2.*wnoise_TT
    nT = modlmap*0.+(args.noise*(np.pi/180./60.))**2.
    nP = 2.*nT
    ncomp = 3 if pol else 1
    ps = np.zeros((ncomp,ncomp,ells.size))
    ps[0,0] = wnoise_TT
    if pol:
        ps[1,1] = wnoise_PP
        ps[2,2] = wnoise_PP
    oshape = (3,)+ishape if pol else ishape
    ngen = maps.MapGen(oshape,iwcs,ps)
    
    tmask = maps.mask_kspace(ishape,iwcs,lmin=args.tellmin,lmax=args.tellmax)
    pmask = maps.mask_kspace(ishape,iwcs,lmin=args.pellmin,lmax=args.pellmax)
    kmask = maps.mask_kspace(ishape,iwcs,lmin=args.kellmin,lmax=args.kellmax)

    qest = lensing.qest(ishape,iwcs,theory,noise2d=nT,beam2d=kbeam,kmask=tmask,noise2d_P=nP,kmask_P=pmask,kmask_K=kmask,pol=pol,grad_cut=None,unlensed_equals_lensed=True)

    taper,w2 = maps.get_taper_deg(ishape,iwcs,taper_width_degrees = args.taper_width,pad_width_degrees = args.pad_width)
    fc = maps.FourierCalc(oshape,iwcs,iau=args.iau)
    
    return qest,ngen,kbeam,binner,taper,fc

def lens(ulensed,convergence):
    posmap = ulensed.posmap()
    kmask = maps.mask_kspace(ulensed.shape,ulensed.wcs,lmin=10,lmax=8000)
    phi,_ = lensing.kappa_to_phi(enmap.enmap(maps.filter_map(enmap.enmap(convergence,wcs),kmask),wcs),ulensed.modlmap(),return_fphi=True)
    grad_phi = enmap.grad(phi)
    pos = posmap + grad_phi
    alpha_pix = ulensed.sky2pix(pos, safe=False)
    lensed = enlensing.displace_map(ulensed, alpha_pix, order=5)
    return lensed


def init_flat(ishape,iwcs):
    modlmap = enmap.modlmap(ishape,iwcs)
    lmax = modlmap.max()
    ells = np.arange(2,lmax,1)
    ncomp = 3 if pol else 1
    ps = np.zeros((ncomp,ncomp,ells.size))
    ps[0,0] = theory.uCl('TT',ells)
    if pol:
        ps[1,1] = theory.uCl('EE',ells)
        ps[1,2] = theory.uCl('TE',ells)
        ps[2,1] = theory.uCl('TE',ells)
    oshape = (3,)+ishape if pol else ishape
    mgen = maps.MapGen(oshape,iwcs,ps)
    psk = theory.gCl('kk',ells).reshape((1,1,ells.size))
    kgen = maps.MapGen(ishape,iwcs,psk)
    return mgen,kgen


inited = False
for i,task in enumerate(my_tasks):


    filename = lambda x: args.SimRoot+"_"+x+"_"+str(task).zfill(4)+".fits"
    try:
        if args.flat and args.flat_force: raise
        cpatch = enmap.read_fits(filename("lensed"),box=box if not(args.flat) else None,wcs_override=fwcs if not(args.flat) else None)
        kpatch = enmap.read_fits(filename("kappa"),box=box if not(args.flat) else None,wcs_override=fwcs if not(args.flat) else None)
        if i==0: shape,wcs = cpatch.shape,cpatch.wcs
    except:
        assert args.flat, "No sims found in directory specified. I can only make lensed sims if they are flat-sky."
        shape,wcs = enmap.geometry(pos=box,res=args.flat_res*np.pi/180./60.)
        if pol: shape = (3,) + shape
        if not(inited): mgen, kgen = init_flat(shape[-2:],wcs)
        inited = True
        unlensed = mgen.get_map(iau=args.iau)
        kpatch = kgen.get_map()
        cpatch = lens(unlensed,kpatch)
        enmap.write_fits(filename("lensed"),cpatch)
        enmap.write_fits(filename("unlensed"),unlensed)
        enmap.write_fits(filename("kappa"),kpatch)
        
    if i==0:
        qest, ngen, kbeam, binner, taper, fc = init_geometry(shape[-2:],wcs)
        if args.flat and not(args.flat_taper): taper = kpatch*0.+1.
        w3 = np.mean(taper**3.)
        w2 = np.mean(taper**2.)

    kpatch *= taper
    
    nmaps = ngen.get_map(iau=args.iau)
    observed = maps.filter_map(cpatch*taper,kbeam) + nmaps*taper
    lteb = fc.iqu2teb(observed,normalize=False)
    lt,le,lb = lteb[0],lteb[1],lteb[2]
    
    p2d = fc.f2power(lt,lt)
    cents,p1d = binner.bin(p2d/w2)
    st.add_to_stats("cTT",p1d.copy())
    p2d = fc.f2power(le,le)
    cents,p1d = binner.bin(p2d/w2)
    st.add_to_stats("cEE",p1d.copy())
    p2d = fc.f2power(lb,lb)
    cents,p1d = binner.bin(p2d/w2)
    st.add_to_stats("cBB",p1d.copy())
    p2d = fc.f2power(lt,le)
    cents,p1d = binner.bin(p2d/w2)
    st.add_to_stats("cTE",p1d.copy())
    

    
    if args.debug and task==0:
        io.plot_img(cpatch[0],io.dout_dir+"cmbI.png",high_res=False)
        if pol:
            io.plot_img(cpatch[1],io.dout_dir+"cmbQ.png",high_res=False)
            io.plot_img(cpatch[2],io.dout_dir+"cmbU.png",high_res=False)
            io.plot_img(cpatch[1],io.dout_dir+"cmbQh.png",high_res=True)
            io.plot_img(cpatch[2],io.dout_dir+"cmbUh.png",high_res=True)
        io.plot_img(nmaps[0],io.dout_dir+"nI.png",high_res=False)
        if pol:
            io.plot_img(nmaps[1],io.dout_dir+"nQ.png",high_res=False)
            io.plot_img(nmaps[2],io.dout_dir+"nU.png",high_res=False)
        io.plot_img(kpatch,io.dout_dir+"kappa.png",high_res=False)

    
    p2d,kinp,kinp = fc.power2d(kpatch)
    cents,p1dii = binner.bin(p2d/w2)
    st.add_to_stats("input",p1dii.copy())
    for pcomb in polcombs:
        recon = qest.kappa_from_map(pcomb,lt,le,lb,alreadyFTed=True)
        if args.debug and task==0: io.plot_img(recon,io.dout_dir+"recon"+pcomb+".png",high_res=False)
        p2d,krecon = fc.f1power(recon,kinp)
        cents,p1d = binner.bin(p2d/w3)
        st.add_to_stats(pcomb,p1d.copy())

        st.add_to_stats("r_"+pcomb,(p1d-p1dii)/p1dii)



        
    if rank==0: print ("Rank 0 done with task ", task+1, " / " , len(my_tasks))

if rank==0: print ("Collecting results...")
st.get_stats(verbose=False)
# st.get_stacks(verbose=False)


if rank==0:



    pl = io.Plotter(yscale='log',xlabel='$L$',ylabel='$C_L$')
    ells = np.arange(2,args.kellmax,1)
    pl.add(ells,theory.gCl('kk',ells),lw=3,color="k")
    pl.add(cents,st.stats['input']['mean'],lw=1,color="k",alpha=0.5)
    for pcomb in polcombs:
        pmean,perr = st.stats[pcomb]['mean'],st.stats[pcomb]['errmean']
        pl.add_err(cents,pmean,yerr=perr,marker="o",mew=2,elinewidth=2,ls="-",lw=2,label=pcomb)

    pl.legend(loc='upper right')
    pl._ax.set_ylim(1e-9,1e-6)
    pl.done(io.dout_dir+"clkk.png")


    pl = io.Plotter(xlabel='$L$',ylabel='$\Delta C_L/C_L$')
    for pcomb in polcombs:
        pmean,perr = st.stats["r_"+pcomb]['mean'],st.stats["r_"+pcomb]['errmean']
        pl.add_err(cents,pmean,yerr=perr,marker="o",mew=2,elinewidth=2,ls="-",lw=2,label=pcomb)
    pl.legend(loc='upper right')
    pl.hline()
    pl._ax.set_ylim(-0.1,0.05)
    pl.done(io.dout_dir+"rclkk.png")



    pl = io.Plotter(yscale='log',xlabel='$L$',ylabel='$C_{\\ell}$')
    ells = np.arange(2,args.pellmax,1)
    for cmb in ['TT','EE','BB']:
        pl.add(ells,theory.lCl(cmb,ells)*ells**2.,lw=3,color="k")
        pmean,perr = st.stats["c"+cmb]['mean'],st.stats["c"+cmb]['errmean']
        pl.add_err(cents,pmean*cents**2.,yerr=perr*cents**2.,marker="o",mew=2,elinewidth=2,ls="-",lw=2,label=cmb)

    pl.legend(loc='upper right')
    # pl._ax.set_ylim(1e-9,1e-6)
    pl.done(io.dout_dir+"clcmb.png")

    
    pl = io.Plotter(xlabel='$L$',ylabel='$C_{\\ell}$')
    ells = np.arange(2,args.pellmax,1)
    pl.add(ells,theory.lCl('TE',ells)*ells**2.,lw=3,color="k")
    pmean,perr = st.stats['cTE']['mean'],st.stats['cTE']['errmean']
    pl.add_err(cents,pmean*cents**2.,yerr=perr*cents**2.,marker="o",mew=2,elinewidth=2,ls="-",lw=2)
    pl.legend(loc='upper right')
    # pl._ax.set_ylim(1e-9,1e-6)
    pl.done(io.dout_dir+"clte.png")
