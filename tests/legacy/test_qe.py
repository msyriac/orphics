from __future__ import print_function
from orphics import maps,io,cosmology,mpi,stats,lensing
from enlib import enmap,lensing as enlensing
import numpy as np
import os,sys
import argparse

"""

Loads cutouts from full-sky lensed simulations or generate flat-sky sims on the fly.
Performs QE lensing reconstruction for both temperature
and polarization estimators.
Cross-correlates with input convergence to verify reconstruction.
(Does not verify auto-spectra)

Full-sky sims work to <5% for all estimators if iau=True. Pol estimators
esp. TB are worse, possibly due to E->B leakage.

Pol estimators for noiseless and lmax=5000 seem to have large N2-like bias.

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
parser.add_argument("--noise-pad",     type=float,  default=3.5,help="Extra degrees to pad noise maps by if using full-sky signal sims.")
parser.add_argument("--tellmin",     type=int,  default=500,help="Temperature ellmin.")
parser.add_argument("--tellmax",     type=int,  default=3000,help="Temperature ellmax.")
parser.add_argument("--pellmin",     type=int,  default=500,help="Polarization ellmin.")
parser.add_argument("--pellmax",     type=int,  default=5000,help="Polarization ellmax.")
parser.add_argument("--kellmin",     type=int,  default=40,help="Convergence ellmin.")
parser.add_argument("--kellmax",     type=int,  default=3000,help="Convergence ellmax.")
parser.add_argument("--dell",     type=int,  default=100,help="Spectra bin-width.")
parser.add_argument("--estimators",     type=str,  default="TT,TE,ET,EB,EE,TB",help="List of polcombs.")
parser.add_argument("--save",     type=str,  default="",help="Suffix for saves.")
parser.add_argument("--save-meanfield",     type=str,  default=None,help="Runs only meanfield. Path to save meanfield to.")
parser.add_argument("--load-meanfield",     type=str,  default=None,help="Path to load meanfields from.")
parser.add_argument("--debug", action='store_true',help='Debug with plots.')
parser.add_argument("--debug-noise", action='store_true',help='Debug noise spectra.')
parser.add_argument("--iau", action='store_true',help='Use IAU pol convention.')
parser.add_argument("--flat", action='store_true',help='Do flat sky periodic.')
parser.add_argument("-f","--flat-force", action='store_true',help='Force flat sky remake.')
parser.add_argument("-p","--purify", action='store_true',help='Purify E/B.')
parser.add_argument("--res",     type=float,  default=1.0,help="Resolution in arcmin of sims.")
parser.add_argument("--flat-taper", action='store_true',help='Taper periodic flat-sky sims.')
#parser.add_argument("-p", "--pad", action='store_true',help='Big if true.')
args = parser.parse_args()
san = lambda x: np.nan_to_num(x)
polcombs = args.estimators.split(',')
pol = False if polcombs==['TT'] else True
io.dout_dir += "qev_"
if args.flat: io.dout_dir += "flat_"
if args.iau: io.dout_dir += "iau"
io.dout_dir += args.save+"_"

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
fshape,fwcs = enmap.fullsky_geometry(res=args.res*np.pi/180./60.)
box = np.array([[-args.wy/2.,-args.wx/2.],[args.wy/2.,args.wx/2.]])*np.pi/180.

# Stats
st = stats.Stats(comm)

def init_geometry(ishape,iwcs):
    modlmap = enmap.modlmap(ishape,iwcs)
    bin_edges = np.arange(args.kellmin,args.kellmax,args.dell)
    binner = stats.bin2D(modlmap,bin_edges)
    if args.beam<1e-5:
        kbeam = None
    else:
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

    if not(args.flat) and args.noise_pad>1.e-5:
        # Pad noise sim geometry
        pad_width_deg = args.noise_pad
        pad_width = pad_width_deg * np.pi/180.
        res = maps.resolution(oshape[-2:],iwcs)
        pad_pixels = int(pad_width/res)
        template = enmap.zeros(oshape,iwcs)
        btemplate = enmap.pad(template,pad_pixels)
        bshape,bwcs = btemplate.shape,btemplate.wcs
        del template
        del btemplate
        ngen = maps.MapGen(bshape,bwcs,ps)
    else:
        ngen = maps.MapGen(oshape,iwcs,ps)
    
    tmask = maps.mask_kspace(ishape,iwcs,lmin=args.tellmin,lmax=args.tellmax)
    pmask = maps.mask_kspace(ishape,iwcs,lmin=args.pellmin,lmax=args.pellmax)
    kmask = maps.mask_kspace(ishape,iwcs,lmin=args.kellmin,lmax=args.kellmax)

    qest = lensing.qest(ishape,iwcs,theory,noise2d=nT,beam2d=kbeam,kmask=tmask,noise2d_P=nP,kmask_P=pmask,kmask_K=kmask,pol=pol,grad_cut=None,unlensed_equals_lensed=True)

    taper,w2 = maps.get_taper_deg(ishape,iwcs,taper_width_degrees = args.taper_width,pad_width_degrees = args.pad_width)
    fc = maps.FourierCalc(oshape,iwcs,iau=args.iau)

    
    purifier = maps.Purify(ishape,iwcs,taper) if args.purify else None

    
    return qest,ngen,kbeam,binner,taper,fc,purifier

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
        ps[1,0] = theory.uCl('TE',ells)
        ps[0,1] = theory.uCl('TE',ells)
    oshape = (3,)+ishape if pol else ishape
    mgen = maps.MapGen(oshape,iwcs,ps)
    psk = theory.gCl('kk',ells).reshape((1,1,ells.size))
    kgen = maps.MapGen(ishape,iwcs,psk)
    return mgen,kgen


mfs = {}
for pcomb in polcombs:
    if not(args.load_meanfield is None):
        mfs[pcomb] = enmap.read_hdf(args.load_meanfield+"_mf_"+pcomb+".hdf")
    else:
        mfs[pcomb] = 0.

inited = False
for i,task in enumerate(my_tasks):


    filename = lambda x,ext="fits",k=task: args.SimRoot+"_"+x+"_"+str(k).zfill(4)+args.save+"."+ext
    try:
        if args.flat and args.flat_force: raise
        if not(args.save_meanfield is None): raise
        cpatch = enmap.read_fits(filename("lensed"),box=box if not(args.flat) else None,wcs_override=fwcs if not(args.flat) else None)
        kpatch = enmap.read_fits(filename("kappa"),box=box if not(args.flat) else None,wcs_override=fwcs if not(args.flat) else None)
        if i==0: shape,wcs = cpatch.shape,cpatch.wcs
    except:
        assert args.flat or not(args.save_meanfield is None), "No sims found in directory specified. I can only make lensed sims if they are flat-sky."
        if not(inited): 
            if not(args.save_meanfield is None) and not(args.flat):
                template = enmap.read_fits(filename("lensed","fits",0),box=box if not(args.flat) else None,wcs_override=fwcs if not(args.flat) else None)
                shape,wcs = template.shape,template.wcs 
            else:
                shape,wcs = enmap.geometry(pos=box,res=args.res*np.pi/180./60., proj="car")
                if pol: shape = (3,) + shape
            mgen, kgen = init_flat(shape[-2:],wcs)
        inited = True
        unlensed = mgen.get_map(iau=args.iau)

        if not(args.save_meanfield is None):
            cpatch = unlensed
        else:
            kpatch = kgen.get_map()
            cpatch = lens(unlensed,kpatch)
        # enmap.write_fits(filename("lensed"),cpatch)
        # enmap.write_fits(filename("unlensed"),unlensed)
        enmap.write_fits(filename("kappa"),kpatch)

    if i==0:
        qest, ngen, kbeam, binner, taper, fc, purifier = init_geometry(shape[-2:],wcs)
        if args.flat and not(args.flat_taper): taper = enmap.ones(shape[-2:])
        w4 = np.mean(taper**4.)
        w3 = np.mean(taper**3.)
        w2 = np.mean(taper**2.)

    if args.save_meanfield is None: kpatch *= taper
    
    nmaps = ngen.get_map(iau=args.iau)
    if not(args.flat) and args.noise_pad>1.e-5:
        nmaps = enmap.extract(nmaps,shape,wcs)

    observed = maps.convolve_gaussian(cpatch,args.beam) + nmaps if args.beam>1e-5 and args.noise>1e-5 else cpatch

    enmap.write_fits(filename("obs_I"),cpatch[0])
    enmap.write_fits(filename("obs_Q"),cpatch[1])
    enmap.write_fits(filename("obs_U"),cpatch[2])

    if args.purify:
        lt,le,lb = purifier.lteb_from_iqu(observed,method='pure') # no need to multiply by window if purifying
        if args.debug_noise: lnt,lne,lnb = purifier.lteb_from_iqu(nmaps,method='pure') # no need to multiply by window if purifying
    else:
        lteb = fc.iqu2teb(observed*taper,normalize=False)
        
        lt,le,lb = lteb[0],lteb[1],lteb[2]
        if args.debug_noise: 
            nlteb = fc.iqu2teb(nmaps*taper,normalize=False)
            lnt,lne,lnb = nlteb[0],nlteb[1],nlteb[2]
    if args.flat and not(args.iau):
        lb = -lb
        if args.debug_noise:  lnb = -lnb

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
    if args.debug_noise:
        p2d = fc.f2power(lnt,lnt)
        cents,p1d = binner.bin(p2d/w2)
        st.add_to_stats("nTT",p1d.copy())
        p2d = fc.f2power(lne,lne)
        cents,p1d = binner.bin(p2d/w2)
        st.add_to_stats("nEE",p1d.copy())
        p2d = fc.f2power(lnb,lnb)
        cents,p1d = binner.bin(p2d/w2)
        st.add_to_stats("nBB",p1d.copy())
        
    

    
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
        if args.save_meanfield is None: io.plot_img(kpatch,io.dout_dir+"kappa.png",high_res=False)

    if args.save_meanfield is None: 
        p2d,kinp,kinp = fc.power2d(kpatch)
        cents,p1dii = binner.bin(p2d/w2)
        st.add_to_stats("input",p1dii.copy())
    te_et_comb = 0.
    te_et_comb_wt = 0.
    mv_comb = 0.
    mv_comb_wt = 0.
    krtt = None
    kreb = None
    
    for pcomb in polcombs:
        recon = qest.kappa_from_map(pcomb,lt,le,lb,alreadyFTed=True)-mfs[pcomb]
        enmap.write_fits(filename("recon_"+pcomb),recon)
        if args.debug and task==0: io.plot_img(recon,io.dout_dir+"recon"+pcomb+".png",high_res=False)
        if args.save_meanfield is None: 
            p2d,krecon = fc.f1power(recon,kinp)
            cents,p1d = binner.bin(p2d/w3)
            st.add_to_stats(pcomb,p1d.copy())
            if pcomb=='TT': krtt=krecon.copy()
            if pcomb=='EB': kreb=krecon.copy()
            # TE + ET
            if pcomb in ['TE','ET']:
                te_et_comb += np.nan_to_num(krecon/qest.N.Nlkk[pcomb])
                te_et_comb_wt += np.nan_to_num(1./qest.N.Nlkk[pcomb])
            # MV
            mv_comb += np.nan_to_num(krecon/qest.N.Nlkk[pcomb])
            mv_comb_wt += np.nan_to_num(1./qest.N.Nlkk[pcomb])
        else:
            st.add_to_stack("mf_"+pcomb,recon)
        if args.save_meanfield is None: st.add_to_stats("r_"+pcomb,(p1d-p1dii)/p1dii)


    # TE + ET
    pcomb = "TE_ET"
    kte_et = te_et_comb/te_et_comb_wt
    p2d = fc.f2power(kte_et,kinp)
    cents,p1d = binner.bin(p2d/w3)
    st.add_to_stats(pcomb,p1d.copy())
    if args.save_meanfield is None: st.add_to_stats("r_"+pcomb,(p1d-p1dii)/p1dii)
    # MV
    pcomb = "mv"
    krecon_mv = mv_comb/mv_comb_wt
    p2d = fc.f2power(krecon_mv,kinp)
    cents,p1d = binner.bin(p2d/w3)
    st.add_to_stats("mv",p1d.copy())
    if args.save_meanfield is None: st.add_to_stats("r_"+pcomb,(p1d-p1dii)/p1dii)
    # TTEB
    if (args.save_meanfield is None) and (krtt is not None) and (kreb is not None):
        p2dtteb = fc.f2power(krtt,kreb)
        cents,p1d = binner.bin(p2dtteb/w4)
        st.add_to_stats("tteb",p1d.copy())
        st.add_to_stats("r_tteb",(p1d-p1dii)/p1dii)
    

        
    if rank==0: print ("Rank 0 done with task ", task+1, " / " , len(my_tasks))

if rank==0: print ("Collecting results...")
if not(args.save_meanfield is None):
    st.get_stacks(verbose=False)
else:
    st.get_stats(verbose=False)


if rank==0:



        
    

    if not(args.save_meanfield is None):
        for pcomb in polcombs:
            pmf = st.stacks["mf_"+pcomb]
            enmap.write_hdf(args.save_meanfield+"_mf_"+pcomb+".hdf",pmf.copy())
            io.plot_img(pmf,io.dout_dir+"mf_"+pcomb+"_highres.png",high_res=True)
            io.plot_img(pmf,io.dout_dir+"mf_"+pcomb+".png",high_res=False)
    else:

        pl = io.Plotter(yscale='log',xlabel='$L$',ylabel='$C_L$')
        ells = np.arange(2,args.kellmax,1)
        pl.add(ells,theory.gCl('kk',ells),lw=3,color="k")
        pl.add(cents,st.stats['input']['mean'],lw=1,color="k",alpha=0.5)
        for pcomb in polcombs+['TE_ET','mv','tteb']:
            pmean,perr = st.stats[pcomb]['mean'],st.stats[pcomb]['errmean']
            pl.add_err(cents,pmean,yerr=perr,marker="o",mew=2,elinewidth=2,lw=2,label=pcomb,ls="--" if pcomb=='mv' else "-")

        pl.legend(loc='upper right')
        pl._ax.set_ylim(1e-9,1e-6)
        pl.done(io.dout_dir+"clkk.png")


        pl = io.Plotter(xlabel='$L$',ylabel='$\Delta C_L/C_L$')
        save_tuples = []
        save_tuples.append(cents)
        header = ""
        header += "L \t"
        for pcomb in polcombs+['TE_ET','mv','tteb']:
            pmean,perr = st.stats["r_"+pcomb]['mean'],st.stats["r_"+pcomb]['errmean']
            pl.add_err(cents,pmean,yerr=perr,marker="o",mew=2,elinewidth=2,lw=2,label=pcomb,ls="--" if pcomb=='mv' else "-")
            save_tuples.append(pmean)
            save_tuples.append(perr)
            header += pcomb+" \t"
            header += pcomb+"_err \t"
        io.save_cols(filename("rclkk","txt"),save_tuples,header=header,delimiter='\t')
        pl.legend(loc='upper right')
        pl.hline()
        pl._ax.set_ylim(-0.1,0.05)
        pl.done(io.dout_dir+"rclkk.png")



        pl = io.Plotter(yscale='log',xlabel='$L$',ylabel='$C_{\\ell}$')
        ells = np.arange(2,args.pellmax,1)
        for i,cmb in enumerate(['TT','EE','BB']):
            pl.add(ells,theory.lCl(cmb,ells)*ells**2.,lw=3,color="k")
            pmean,perr = st.stats["c"+cmb]['mean'],st.stats["c"+cmb]['errmean']
            pl.add_err(cents,pmean*cents**2.,yerr=perr*cents**2.,marker="o",mew=2,elinewidth=2,ls="-",lw=2,label=cmb,color="C"+str(i))
            if args.debug_noise:
                pmean,perr = st.stats["n"+cmb]['mean'],st.stats["n"+cmb]['errmean']
                pl.add(cents,pmean*cents**2.,ls="--",lw=2,color="C"+str(i))
                q2d = qest.N.noiseYY2d[cmb]
                cents,p1d = binner.bin(q2d.real)
                pl.add(cents,p1d*cents**2.,ls="-",lw=2,color="C"+str(i))


        pl.legend(loc='upper right')
        # pl._ax.set_ylim(1e-9,1e-6)
        pl.done(io.dout_dir+"clcmb.png")

        if args.debug_noise:

            pl = io.Plotter(yscale='log',xlabel='$L$',ylabel='$C_{\\ell}$')
            ells = np.arange(2,args.pellmax,1)
            for i,cmb in enumerate(['TT','EE','BB']):
                pmean,perr = st.stats["n"+cmb]['mean'],st.stats["n"+cmb]['errmean']
                pl.add(cents,pmean*cents**2.,ls="none",lw=2,color="C"+str(i),marker="o")
                q2d = qest.N.noiseYY2d[cmb]
                cents,p1d = binner.bin(q2d.real)
                pl.add(cents,p1d*cents**2.,ls="-",lw=2,color="C"+str(i))

            pl.legend(loc='upper right')
            # pl._ax.set_ylim(1e-9,1e-6)
            pl.done(io.dout_dir+"nlcmb.png")

        

        pl = io.Plotter(xlabel='$L$',ylabel='$C_{\\ell}$')
        ells = np.arange(2,args.pellmax,1)
        pl.add(ells,theory.lCl('TE',ells)*ells**2.,lw=3,color="k")
        pmean,perr = st.stats['cTE']['mean'],st.stats['cTE']['errmean']
        pl.add_err(cents,pmean*cents**2.,yerr=perr*cents**2.,marker="o",mew=2,elinewidth=2,ls="-",lw=2)
        pl.legend(loc='upper right')
        # pl._ax.set_ylim(1e-9,1e-6)
        pl.done(io.dout_dir+"clte.png")
