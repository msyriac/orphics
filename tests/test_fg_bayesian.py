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
parser.add_argument("-a", "--arc",     type=float,  default=10.,help="Stamp width (arcmin).")
parser.add_argument("-p", "--pix",     type=float,  default=0.5,help="Pix width (arcmin).")
parser.add_argument("-n", "--noise",     type=float,  default=1.0,help="Noise (uK-arcmin).")
parser.add_argument("-f", "--foregrounds",     type=str,  default="-1,1,20,0.2",help="Foreground amplitudes specified as a list fmin,fmax,fnum,ftrue.")
#parser.add_argument("-f", "--flag", action='store_true',help='A flag.')
args = parser.parse_args()
fmin,fmax,fnum,ftrue = [float(x) for x in args.foregrounds.split(',')]

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
pout_dir = PathConfig.get("paths","plots")+args.GridName+"/joint_bayesian_hdv_plots_"+io.join_nums((barc,bpix,bbeam,args.noise))+"_"
io.mkdir(pout_dir,comm)


# Tiny Geometry
bshape, bwcs = maps.rect_geometry(width_arcmin=barc,px_res_arcmin=bpix,pol=False)
bmodlmap = enmap.modlmap(bshape,bwcs)
bmodrmap = enmap.modrmap(bshape,bwcs)



if rank==0:
    print(bshape,bwcs)



# Noise model
noise_uK_rad = args.noise*np.pi/180./60.
normfact = np.sqrt(np.prod(enmap.pixsize(bshape,bwcs)))
noise_uK_pixel = noise_uK_rad/normfact
Ncov = np.diag([(noise_uK_pixel)**2.]*np.prod(bshape))

kbeam = maps.gauss_beam(bbeam,bmodlmap)

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
            cinv = np.linalg.inv(Tcov).astype(np.float64)


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


# Theory
theory_file_root = "../alhazen/data/Aug6_highAcc_CDM"
cc = counts.ClusterCosmology(skipCls=True)
theory = cosmology.loadTheorySpectraFromCAMB(theory_file_root,unlensedEqualsLensed=False,
                                             useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=False)


# Simulate
lmax = int(bmodlmap.max()+1)
ells = np.arange(0,lmax,1)
ps = theory.uCl('TT',ells).reshape((1,1,lmax))
ps_noise = np.array([(noise_uK_rad)**2.]*ells.size).reshape((1,1,ells.size))
mg = maps.MapGen(bshape,bwcs,ps)
ng = maps.MapGen(bshape,bwcs,ps_noise)
kamp_true = args.Amp

kappa = lensing.nfw_kappa(kamp_true*1e15,bmodrmap,cc,overdensity=200.,critical=True,atClusterZ=True)


phi,_ = lensing.kappa_to_phi(kappa,bmodlmap,return_fphi=True)
grad_phi = enmap.grad(phi)
posmap = enmap.posmap(bshape,bwcs)
pos = posmap + grad_phi
alpha_pix = enmap.sky2pix(bshape,bwcs,pos, safe=False)
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

for i,task in enumerate(my_tasks):
    if (i+1)%10==0 and rank==0: print(i+1)

    # Sim
    unlensed = mg.get_map()
    noise_map = ng.get_map()
    lensed = enlensing.displace_map(unlensed, alpha_pix, order=lens_order)
    fg = kappa * 50. * ftrue
    tot_beamed = maps.filter_map(lensed+fg,kbeam)
    stamp = tot_beamed  + noise_map
    if task==0:
        io.plot_img(unlensed,pout_dir + "0_unlensed.png")
        io.plot_img(lensed,pout_dir + "1_lensed.png")
        io.plot_img(fg,pout_dir + "2_fg.png")
        io.plot_img(stamp,pout_dir + "2_tot.png")

    # Bayesian

    totlnlikes = []    
    for k,kamp in enumerate(bkamps):
        lnlike = maps.get_lnlike(cinvs[k],stamp) + logdets[k]
        totlnlike = lnlike #+ lnprior[k]
        totlnlikes.append(totlnlike)
    nlnlikes = -0.5*np.array(totlnlikes)
    mstats.add_to_stats("totlikes",nlnlikes)


mstats.get_stats(verbose=False)
mstats.get_stacks(verbose=False)

if rank==0:
    # Bayesian
    blnlikes = mstats.vectors["totlikes"].sum(axis=0)
    blnlikes -= blnlikes.max()



    
    pl1 = io.Plotter(xlabel="$A$",ylabel="$\\mathrm{ln}\\mathcal{L}$")
    

    pl1.add(bkamps,blnlikes,label="bayesian chisquare")

    bamaxes = bkamps[np.isclose(blnlikes,blnlikes.max())]
    bp = np.polyfit(bkamps,blnlikes,2)
    pl1.add(bkamps,bp[0]*bkamps**2.+bp[1]*bkamps+bp[2],ls="--",label="bayesian chisquare fit")
    for bamax in bamaxes:
        pl1.vline(x=bamax,ls="-")

    pl1.vline(x=kamp_true,ls="--")
    pl1.legend(loc='upper left')
    pl1.done(pout_dir+"lensed_lnlikes_all.png")
    
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
    
    pl2 = io.Plotter(xlabel="$A$",ylabel="$\\mathcal{L}$")
    
    pl2.add(nkamps,np.exp(-(nkamps-mean)**2./2./sigma**2.),label="BE likelihood from chisquare fit")
    # pl2.add(bkamps,like,label="BE likelihood")


    pl2.vline(x=kamp_true,ls="--")
    pl2.legend(loc='upper left')
    pl2.done(pout_dir+"lensed_likes.png")

    
