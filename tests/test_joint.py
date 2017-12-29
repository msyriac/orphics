from __future__ import print_function
from orphics import maps,io,cosmology,lensing,stats,mpi
from enlib import enmap,lensing as enlensing,bench,resample
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
parser.add_argument("-a", "--arc",     type=float,  default=100.,help="Stamp width (arcmin).")
parser.add_argument("-p", "--pix",     type=float,  default=0.1953125,help="Pix width (arcmin).")
parser.add_argument("-b", "--beam",     type=float,  default=1.0,help="Beam (arcmin).")
parser.add_argument("-n", "--noise",     type=float,  default=3.0,help="Noise (uK-arcmin).")
parser.add_argument("-m", "--hdv", action='store_true',help='Use HDV mass definition.')
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
barc = attribs['arc'] ; bpix = attribs['pix']  ; bbeam = attribs['beam']  
pout_dir = PathConfig.get("paths","plots")+args.GridName+"/joint_bayesian_hdv_plots_"+io.join_nums((barc,bpix,args.beam,args.noise))+"_"
io.mkdir(pout_dir,comm)


# Tiny Geometry
bshape, bwcs = maps.rect_geometry(width_arcmin=barc,px_res_arcmin=bpix,pol=False)
bmodlmap = enmap.modlmap(bshape,bwcs)
bmodrmap = enmap.modrmap(bshape,bwcs)


# Big Geometry
shape, wcs = maps.rect_geometry(width_arcmin=args.arc,px_res_arcmin=args.pix,pol=False)
modlmap = enmap.modlmap(shape,wcs)
modrmap = enmap.modrmap(shape,wcs)
oshape,owcs = enmap.scale_geometry(shape,wcs,args.pix/bpix)
omodlmap = enmap.modlmap(oshape,owcs)
omodrmap = enmap.modrmap(oshape,owcs)

if rank==0:
    print(bshape,bwcs)
    print(shape,wcs)
    print(oshape,owcs)



# Binning
bin_edges = np.arange(0.,20.0,bpix*2)
binner = stats.bin2D(omodrmap*60.*180./np.pi,bin_edges)

# Noise model
noise_uK_rad = args.noise*np.pi/180./60.
normfact = np.sqrt(np.prod(enmap.pixsize(bshape,bwcs)))
noise_uK_pixel = noise_uK_rad/normfact
Ncov = np.diag([(noise_uK_pixel)**2.]*np.prod(bshape))

kbeam = maps.gauss_beam(args.beam,modlmap)
okbeam = maps.gauss_beam(args.beam,omodlmap)

if True: #True:
    # Load covs
    bkamps = np.loadtxt(GridName+"/amps.txt",unpack=True) #[:1] # !!!
    if rank==0: print("Amplitudes: ",bkamps)
    cov_file = lambda x: GridName+"/cov_"+str(x)+".npy"
    cinvs = []
    logdets = []
    for k in range(len(bkamps)):
        if rank==0:

            print("Loading cov",k," / ",len(bkamps),"...")
            cov = np.load(cov_file(k))

            if np.abs(bbeam-args.beam)/args.beam>1.e-5:
                if k==0:
                    print("Readjusting beam from ", bbeam," in cov to ", args.beam , " specified on command line.")
                    obeam = maps.gauss_beam(bbeam,bmodlmap)
                    nbeam = maps.gauss_beam(args.beam,bmodlmap)
                    beam_ratio = np.nan_to_num(nbeam/obeam)
                cov = lensing.beam_cov(cov,beam_ratio)

            try:
                old_cores = os.environ["OMP_NUM_THREADS"]
            except:
                old_cores = "1"
            import multiprocessing
            num_cores= str(multiprocessing.cpu_count())
            os.environ["OMP_NUM_THREADS"] = num_cores

            Tcov = cov + Ncov + 5000 # !!!
            with bench.show("covwork"):
                s,logdet = np.linalg.slogdet(Tcov)
                assert s>0
                cinv = pinv2(Tcov).astype(np.float64)

            
            os.environ["OMP_NUM_THREADS"] = old_cores

            
            for core in range(1,numcores):
                comm.Send([cinv, mpi.MPI.DOUBLE], dest=core, tag=77)
                comm.send(logdet, dest=core, tag=88)
        else:
            cinv = np.empty((np.prod(bshape),np.prod(bshape)), dtype=np.float64)
            comm.Recv([cinv, mpi.MPI.DOUBLE], source=0, tag=77)
            logdet = comm.recv(source=0, tag=88)
        cinvs.append(cinv)
        logdets.append(logdet)

# else:
#     bkamps = []
#     import glob
#     import cPickle as pickle
#     fs = glob.glob(PathConfig.get("paths","output_data")+"dump/cdump_*")
#     nfs = len(fs)

#     for i in range(nfs):
#         pickle.load(open(PathConfig.get("paths","output_data")+"dump/cdump_*"

# Theory
theory_file_root = "../alhazen/data/Aug6_highAcc_CDM"
cc = counts.ClusterCosmology(skipCls=True)
theory = cosmology.loadTheorySpectraFromCAMB(theory_file_root,unlensedEqualsLensed=False,
                                             useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=False)


# Simulate
lmax = int(modlmap.max()+1)
ells = np.arange(0,lmax,1)
ps = theory.uCl('TT',ells).reshape((1,1,lmax))
ps_noise = np.array([(noise_uK_rad)**2.]*ells.size).reshape((1,1,ells.size))
mg = maps.MapGen(shape,wcs,ps)
ng = maps.MapGen(oshape,owcs,ps_noise)
kamp_true = args.Amp

if args.hdv:
    kappa = lensing.nfw_kappa(kamp_true*1e15,modrmap,cc,overdensity=180.,critical=False,atClusterZ=False)
else:
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
tellmin = omodlmap[omodlmap>2].min(); tellmax = 8000; kellmin = tellmin ; kellmax = 8096
tmask = maps.mask_kspace(oshape,owcs,lmin=tellmin,lmax=tellmax)
kmask = maps.mask_kspace(oshape,owcs,lmin=kellmin,lmax=kellmax)
qest = lensing.qest(oshape,owcs,theory,noise2d=okbeam*0.+(noise_uK_rad)**2.,beam2d=okbeam,kmask=tmask,kmask_K=kmask,pol=False,grad_cut=2000,unlensed_equals_lensed=False)

for i,task in enumerate(my_tasks):
    if (i+1)%10==0 and rank==0: print(i+1)

    # Sim
    unlensed = mg.get_map()
    noise_map = ng.get_map()
    lensed = maps.filter_map(enlensing.displace_map(unlensed, alpha_pix, order=lens_order),kbeam)
    fdownsampled = enmap.enmap(resample.resample_fft(lensed,oshape),owcs)
    stamp = fdownsampled  + noise_map

    # Bayesian
    cutout = stamp[int(oshape[0]/2.-bshape[0]/2.):int(oshape[0]/2.+bshape[0]/2.),int(oshape[0]/2.-bshape[0]/2.):int(oshape[0]/2.+bshape[0]/2.)]

    totlnlikes = []    
    for k,kamp in enumerate(bkamps):
        lnlike = maps.get_lnlike(cinvs[k],cutout) + logdets[k]
        totlnlike = lnlike #+ lnprior[k]
        totlnlikes.append(totlnlike)
    nlnlikes = -0.5*np.array(totlnlikes)
    mstats.add_to_stats("totlikes",nlnlikes)

    # QE
    if task==0:
        io.plot_img(stamp,pout_dir+"cmb_noisy.png")

    recon = qest.kappa_from_map("TT",stamp)
    cents, recon1d = binner.bin(recon)

    mstats.add_to_stats("recon1d",recon1d)
    mstats.add_to_stack("recon",recon)

mstats.get_stats()
mstats.get_stacks()

if rank==0:
    # Bayesian
    blnlikes = mstats.vectors["totlikes"].sum(axis=0)
    blnlikes -= blnlikes.max()

    fit_qe = True # !!!
    # QE
    stack = mstats.stacks['recon'] 
    recon1d = mstats.stats['recon1d']['mean'] 
    recon1d_err = mstats.stats['recon1d']['errmean'] 
    recon1d_cov = mstats.stats['recon1d']['covmean']
    io.plot_img(stack,pout_dir+"stack.png")
    io.plot_img(stack,pout_dir+"stack_hres.png",high_res=True)

    
    
    if args.hdv:
        kappa_true = maps.filter_map(lensing.nfw_kappa(kamp_true*1e15,omodrmap,cc,overdensity=180.,critical=False,atClusterZ=False),kmask) # !!!!
    else:
        kappa_true = maps.filter_map(lensing.nfw_kappa(kamp_true*1e15,omodrmap,cc,overdensity=200.,critical=True,atClusterZ=True),kmask)
        
    cents, ktrue1d = binner.bin(kappa_true)
    
    arcs,ks = np.loadtxt("input/hdv.csv",unpack=True,delimiter=",")
    
    pl = io.Plotter()
    pl.add(cents,ktrue1d,lw=2,color="k",label="true")
    pl.add(arcs,ks,lw=2,alpha=0.5,label="hdv profile")
    pl.add_err(cents,recon1d,recon1d_err,ls="--",label="recon")
    pl.hline()
    pl.legend(loc='upper right')
    pl._ax.set_ylim(-0.01,0.2)
    pl.done(pout_dir+"recon1d.png")


    
    pl1 = io.Plotter(xlabel="$A$",ylabel="$\\mathrm{ln}\\mathcal{L}$")
    pl2 = io.Plotter(xlabel="$A$",ylabel="$\\mathcal{L}$")
    

    arcmaxes = [10.0] #[5.0,7.5,10.0,15.0,20.0,30.0]
    for j,arcmax in enumerate(arcmaxes):
        if fit_qe:

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
                    if args.hdv:
                        template = lensing.nfw_kappa(amp*1e15,omodrmap,cc,overdensity=180.,critical=False,atClusterZ=False) # !!!
                    else:
                        template = lensing.nfw_kappa(amp*1e15,omodrmap,cc,overdensity=200.,critical=True,atClusterZ=True)
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

            p = np.polyfit(kamps,lnlikes,2)

            c,b,a = p
            mean = -b/2./c
            sigma = np.sqrt(-1./2./c)
            print(mean,sigma)
            sn = (kamp_true/sigma)
            print ("S/N fit for 1000 : ",sn*np.sqrt(1000./args.Nclusters))
            pbias = (mean-kamp_true)*100./kamp_true
            #print ("Bias : ",pbias, " %")
            #print ("Bias : ",(mean-kamp_true)/sigma, " sigma")



            # Fit data for bias
            lnlikes = []
            for k1d in k1ds:

                diff = (k1d-recon1d)[cents<=arcmax] 
                lnlike = -0.5*np.dot(np.dot(diff.T,cinv),diff)
                lnlikes.append(lnlike)

            lnlikes = np.array(lnlikes)
            pl1.add(kamps,lnlikes,label="qe chisquare")
            amaxes = kamps[np.isclose(lnlikes,lnlikes.max())]
            p = np.polyfit(kamps,lnlikes,2)
            
            pl1.add(kamps,p[0]*kamps**2.+p[1]*kamps+p[2],ls="--",label="qe chisquare fit")
            for amax in amaxes:
                pl1.vline(x=amax,ls="-")
            
        pl1.add(bkamps,blnlikes,label="bayesian chisquare")

        bamaxes = bkamps[np.isclose(blnlikes,blnlikes.max())]
        bp = np.polyfit(bkamps,blnlikes,2)
        pl1.add(bkamps,bp[0]*bkamps**2.+bp[1]*bkamps+bp[2],ls="--",label="bayesian chisquare fit")
        for bamax in bamaxes:
            pl1.vline(x=bamax,ls="-")

        if fit_qe:
            # QE
            c,b,a = p
            mean = -b/2./c
            sigma = np.sqrt(-1./2./c)
            print(mean,sigma)
            sn = (kamp_true/sigma)
            pbias = (mean-kamp_true)*100./kamp_true
            print ("QE Bias : ",pbias, " %")
            print ("QE Bias : ",(mean-kamp_true)/sigma, " sigma")

            like = np.exp(lnlikes)
            like /= like.max()
            nkamps = np.linspace(kamps.min(),kamps.max(),1000)
            pl2.add(nkamps,np.exp(-(nkamps-mean)**2./2./sigma**2.),label="QE likelihood from chisquare fit")
            pl2.add(kamps,like,label="QE likelihood",alpha=0.2)

        
        # Bayesian
        c,b,a = bp
        mean = -b/2./c
        sigma = np.sqrt(-1./2./c)
        print(mean,sigma)
        sn = (kamp_true/sigma)
        pbias = (mean-kamp_true)*100./kamp_true
        print ("BE Bias : ",pbias, " %")
        print ("BE Bias : ",(mean-kamp_true)/sigma, " sigma")
        print ("S/N for 1000 : ",sn*np.sqrt(1000./args.Nclusters))

        like = np.exp(blnlikes)
        like /= like.max()
        nkamps = np.linspace(bkamps.min(),bkamps.max(),1000)
        pl2.add(nkamps,np.exp(-(nkamps-mean)**2./2./sigma**2.),label="BE likelihood from chisquare fit")
        #pl2.add(bkamps,like,label="BE likelihood")


    pl2.vline(x=kamp_true,ls="--")
    pl2.legend(loc='upper left')
    pl2.done(pout_dir+"lensed_likes.png")

    pl1.vline(x=kamp_true,ls="--")
    pl1.legend(loc='upper left')
    pl1.done(pout_dir+"lensed_lnlikes_all.png")
    
