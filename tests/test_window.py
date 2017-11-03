import numpy as np
import sys, os, glob
from orphics.analysis.pipeline import mpi_distribute, MPIStats
import orphics.tools.stats as stats
import alhazen.io as aio
import orphics.tools.io as io
import orphics.analysis.flatMaps as fmaps
import warnings
import logging
logger = logging.getLogger()
with io.nostdout():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from enlib import enmap, lensing, resample
from alhazen.quadraticEstimator import Estimator
import alhazen.lensTools as lt
from configparser import SafeConfigParser 
from szar.counts import ClusterCosmology
import enlib.fft as fftfast
import argparse

# Runtime params that should be moved to command line
cosmology_section = "cc_erminia"
expf_name = "experiment_noiseless"

# Parse command line
parser = argparse.ArgumentParser(description='Verify lensing reconstruction.')
parser.add_argument("Region", type=str,help='equator/south')
args = parser.parse_args()
region = args.Region

analysis_section = "analysis_sigurd_"+region
rank = 0

# i/o directories
out_dir = os.environ['WWW']+"plots/distsims_"+region+"_"  # for plots


print("Reading config...")

# Read config
iniFile = "../halofg/input/recon.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

print("Params...")

pol = False
shape_dat, wcs_dat = aio.enmap_from_config_section(Config,analysis_section,pol=pol)
analysis_resolution =  np.min(enmap.extent(shape_dat,wcs_dat)/shape_dat[-2:])*60.*180./np.pi
min_ell = fmaps.minimum_ell(shape_dat,wcs_dat)
if rank==0: print("Ell bounds...")

lb = aio.ellbounds_from_config(Config,"reconstruction_sigurd",min_ell)
tellmin = lb['tellminY']
tellmax = lb['tellmaxY']
pellmin = lb['pellminY']
pellmax = lb['pellmaxY']
kellmin = lb['kellmin']
kellmax = lb['kellmax']

if rank==0: print("Patches data...")

parray_dat = aio.patch_array_from_config(Config,expf_name,shape_dat,wcs_dat,dimensionless=True)

if rank==0: print("Attributes...")

lxmap_dat,lymap_dat,modlmap_dat,angmap_dat,lx_dat,ly_dat = fmaps.get_ft_attributes_enmap(shape_dat,wcs_dat)

if rank==0: print("Binners...")

lbin_edges = np.arange(kellmin,kellmax,200)
lbinner_dat = stats.bin2D(modlmap_dat,lbin_edges)

if rank==0: print("Cosmology...")

# === COSMOLOGY ===
theory, cc, lmax = aio.theory_from_config(Config,cosmology_section)
parray_dat.add_theory(cc,theory,lmax)

    
taper_percent = 14.0
pad_percent = 2.0
Ny,Nx = shape_dat
taper = fmaps.cosineWindow(Ny,Nx,lenApodY=int(taper_percent*min(Ny,Nx)/100.),lenApodX=int(taper_percent*min(Ny,Nx)/100.),padY=int(pad_percent*min(Ny,Nx)/100.),padX=int(pad_percent*min(Ny,Nx)/100.))
w2 = np.mean(taper**2.)
w3 = np.mean(taper**3.)
w4 = np.mean(taper**4.)
if rank==0:
    io.quickPlot2d(taper,out_dir+"taper.png")
    print(("w2 : " , w2))

px_dat = analysis_resolution


Nsims = 10
avg = 0.
for i in range(Nsims):
    print(i)
    cmb = parray_dat.get_unlensed_cmb(seed=i)
    ltt2d = fmaps.get_simple_power_enmap(cmb*taper)
    ccents,ltt = lbinner_dat.bin(ltt2d)/w2
    avg += ltt

ltt = avg/Nsims
iltt2d = theory.uCl("TT",parray_dat.modlmap)
ccents,iltt = lbinner_dat.bin(iltt2d)



pl = io.Plotter()

pdiff = (ltt-iltt)/iltt

pl.add(ccents+50,pdiff,marker="o",ls="-")
pl.legendOn(labsize=10,loc="lower left")
pl._ax.axhline(y=0.,ls="--",color="k")
pl._ax.set_ylim(-0.1,0.1)
pl.done(out_dir+"testwindow_clttpdiff.png")



pl = io.Plotter(scaleY='log',scaleX='log')

pl.add(ccents,iltt*ccents**2.)
pl.add(ccents,ltt*ccents**2.,marker="o",ls="none",label="lensed")

pl.legendOn(labsize=10)
pl.done(out_dir+"testwindow_clttp.png")
