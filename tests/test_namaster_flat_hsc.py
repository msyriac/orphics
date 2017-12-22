from __future__ import print_function
from orphics import maps,io,cosmology,mpi,stats,catalogs
from enlib import enmap
import numpy as np
import os,sys,cPickle as pickle,yaml
import pymaster as nmt
import argparse
from scipy.interpolate import interp1d

# Parse command line
parser = argparse.ArgumentParser(description='Verify flat-sky power spectra with an HSC mask.')
parser.add_argument("field", type=str,help='Field name.')
parser.add_argument("-N", "--Nsims",     type=int,  default=100,help="Number of sims.")
parser.add_argument("-m", "--mask", action='store_true',help='Apply the mask.')
parser.add_argument("-a", "--apodization", action='store_true',help='Apodize the mask.')
parser.add_argument("-e", "--purify-e", action='store_true',help='Purify E.')
parser.add_argument("-b", "--purify-b", action='store_true',help='Purify B.')
parser.add_argument("-w", "--apod-width",     type=float,  default=2.0,help="Apodization width in arcminutes.")
parser.add_argument("-t", "--apod-type",     type=str,  default="C1",help="Apodization type C1,C2,Smooth.")
parser.add_argument("-i", "--iau-convention", action='store_true',help='Use IAU convention for polarization.')
parser.add_argument("-o", "--output-dir",     type=str,  default="/gpfs01/astro/workarea/msyriac/data/hsc/wl/output/",
                    help="Output directory with saved masks, etc..")
args = parser.parse_args()

# Apodization
do_mask = args.mask
do_apod = args.apodization
purify_e = args.purify_e
purify_b = args.purify_b
apod_width = args.apod_width/60.
apotype = args.apod_type

# IAU convention
iau = args.iau_convention


# Signal

ellrange_file = "output/ellrange.pkl" ; clkk_file = "output/clkk.pkl" ; clkg_file = "output/clkg.pkl" ; clgg_file = "output/clgg.pkl"
try:
    # Load saved Cls
    ellrange = pickle.load(open(ellrange_file,'r'))
    clkk = pickle.load(open(clkk_file,'r'))
    clkg = pickle.load(open(clkg_file,'r'))
    clgg = pickle.load(open(clgg_file,'r'))
    print("Loaded pickled Cls.")
except:
    # Calculate Limber approximate Cls
    lc = cosmology.LimberCosmology(lmax=3000,pickling=True)
    # With an HSC-like dndz
    z_edges = np.arange(0.,3.0,0.1)
    zcents = (z_edges[1:]+z_edges[:-1])/2.
    lc.addNz(tag="g",zedges=z_edges,nz=catalogs.dndz(zcents))

    # Get Cls
    ellrange = np.arange(0,3000,1)
    lc.generateCls(ellrange)
    clkk = lc.getCl("cmb","cmb")
    clkg = lc.getCl("cmb","g")
    clgg = lc.getCl("g","g")
    pickle.dump(ellrange,open(ellrange_file,'w'))
    pickle.dump(clkk,open(clkk_file,'w'))
    pickle.dump(clkg,open(clkg_file,'w'))
    pickle.dump(clgg,open(clgg_file,'w'))


# Put power spectra in enlib friendly form
ps = np.zeros((3,3,ellrange.size))
ps[0,0] = clkk
ps[1,1] = clgg
ps[2,2] = clgg*0.
ps[0,1] = clkg
ps[1,0] = clkg

# Load saved masks
field = args.field
out_dir = args.output_dir
mask = enmap.read_map(out_dir+args.field.upper()+"_mask_2arc.hdf")

# What is the geometry of this field?
shape, wcs = mask.shape,mask.wcs
shape = (3,)+shape

# Set up a GRF generator
mg = maps.MapGen(shape,wcs,ps)
# Get wavenumbers
modlmap = enmap.modlmap(shape,wcs)
print("Lmax: ", modlmap.max())
print("Unmasked area sq.deg.: ", mask.area()*(180./np.pi)**2.)

# Set up a power-spectrum-doer
fc = maps.FourierCalc(shape,wcs,iau=iau)

# And a binner
bin_edges = np.arange(40,3000,100)
binner = stats.bin2D(modlmap,bin_edges)
binit = lambda x: binner.bin(x)[1]
cents = binner.centers
ells_coupled = cents

# === ENMAP TO NAMASTER ===
# get the extent and shape of our geometry
Ly,Lx = enmap.extent(shape,wcs)
Ny, Nx = shape[-2:]


# Mask fraction
frac = np.sum(mask)*1./mask.size
area = enmap.area(shape,wcs)*frac*(180./np.pi)**2.
print ("Masked area sq.deg.: ", area)

if not(do_mask):
    mask=mask*0.+1.
else:
    if do_apod:
        nmt.mask_apodization_flat(mask,Lx,Ly,aposize=apod_width,apotype=apotype)
w2 = np.mean(mask**2.)  # Naive power scaling factor

N = args.Nsims
s = stats.Stats() # Stats collector

for i in range(N):
    if (i+1)%10==0: print(i+1)

    # Get sim maps (npol,ny,nx) array
    imaps = mg.get_map(scalar=False,iau=iau)*mask



    if i==0:

        # Plots of mask and fields
        io.plot_img(mask,io.dout_dir+field+"_mask.png",high_res=True)
        # io.plot_img(imaps[0],io.dout_dir+field+"_I.png",high_res=True)
        # io.plot_img(imaps[1],io.dout_dir+field+"_Q.png",high_res=True)
        # io.plot_img(imaps[2],io.dout_dir+field+"_U.png",high_res=True)
        # io.plot_img(imaps[0],io.dout_dir+field+"_I_lowres.png")
        # io.plot_img(imaps[1],io.dout_dir+field+"_Q_lowres.png")
        # io.plot_img(imaps[2],io.dout_dir+field+"_U_lowres.png")


        # Quick power sanity check
        p2d,_,_ = fc.power2d(imaps[0],imaps[0])
        # io.plot_img(np.fft.fftshift(np.log10(p2d)),io.dout_dir+"p2d.png")
        p1d = binit(p2d)/w2
        pl = io.Plotter(yscale='log')
        pl.add(ellrange,clkk,lw=2,color="k")
        pl.add(cents,p1d,marker="o")
        pl.done(io.dout_dir+"cls.png")






    # ENMAP TO NAMASTER FIELDS
    mpt,mpq,mpu = imaps[0],imaps[1],imaps[2]
    f0=nmt.NmtFieldFlat(Lx,Ly,mask,[mpt])
    f2=nmt.NmtFieldFlat(Lx,Ly,mask,[mpq,mpu],purify_e=purify_e,purify_b=purify_b)
    # ells_coupled=f0.get_ell_sampling()

    #Bins:
    l0_bins= bin_edges[:-1]
    lf_bins= bin_edges[1:]
    b=nmt.NmtBinFlat(l0_bins,lf_bins)
    ells_uncoupled=b.get_effective_ells()

    if i==0:
        print("Workspaces; calculating coupling matrices")

        w00=nmt.NmtWorkspaceFlat();
        w02=nmt.NmtWorkspaceFlat();
        w22=nmt.NmtWorkspaceFlat();

        wexists = os.path.isfile(out_dir+field+"_w00_flat.dat") and False
        if wexists:
            w00.read_from(out_dir+field+"_w00_flat.dat")
            w02.read_from(out_dir+field+"_w02_flat.dat")
            w22.read_from(out_dir+field+"_w22_flat.dat")
            print("Loaded saved workspaces.")
        else:
            w00.compute_coupling_matrix(f0,f0,b)
            w02.compute_coupling_matrix(f0,f2,b)
            w22.compute_coupling_matrix(f2,f2,b)


            w00.write_to(out_dir+field+"_w00_flat.dat"); 
            w02.write_to(out_dir+field+"_w02_flat.dat"); 
            w22.write_to(out_dir+field+"_w22_flat.dat"); 

        

    #Computing power spectra:
    cl00_coupled=nmt.compute_coupled_cell_flat(f0,f0); cl00_uncoupled=w00.decouple_cell(cl00_coupled)
    cl02_coupled=nmt.compute_coupled_cell_flat(f0,f2); cl02_uncoupled=w02.decouple_cell(cl02_coupled)
    cl22_coupled=nmt.compute_coupled_cell_flat(f2,f2); cl22_uncoupled=w22.decouple_cell(cl22_coupled)

    # Collect statistics
    s.add_to_stats("ukk",cl00_uncoupled[0])
    s.add_to_stats("ckk",cl00_coupled[0]/w2)
    s.add_to_stats("uke",-cl02_uncoupled[0])   # ENLIB CONVENTION GIVES WRONG SIGN OF TE
    s.add_to_stats("cke",-cl02_coupled[0]/w2)
    s.add_to_stats("uee",cl22_uncoupled[0])
    s.add_to_stats("cee",cl22_coupled[0]/w2)
    
    s.add_to_stats("ukb",cl02_uncoupled[1])
    s.add_to_stats("ueb",cl22_uncoupled[1])
    s.add_to_stats("ube",cl22_uncoupled[2])
    s.add_to_stats("ubb",cl22_uncoupled[3])
    s.add_to_stats("ckb",cl02_coupled[1]/w2)
    s.add_to_stats("ceb",cl22_coupled[1]/w2)
    s.add_to_stats("cbe",cl22_coupled[2]/w2)
    s.add_to_stats("cbb",cl22_coupled[3]/w2)


    # Naive PS calculation
    p2d,_,_ = fc.power2d(imaps,imaps)
    mtt = binit(p2d[0,0])
    mte = binit(p2d[0,1])
    mee = binit(p2d[1,1])
    s.add_to_stats("mtt",mtt)
    s.add_to_stats("mee",mee)
    s.add_to_stats("mte",mte)
    
    

s.get_stats()

# PLOT STUFF

io.plot_img(stats.cov2corr(s.stats["uke"]['cov']),io.dout_dir+"cov.png",flip=False)
io.plot_img(stats.cov2corr(s.stats["cke"]['cov']),io.dout_dir+"ccov.png",flip=False)
def gstats(key):
    return s.stats[key]['mean'],s.stats[key]['errmean']
    
pl = io.Plotter(yscale='log',xlabel="$L$",ylabel="$C_L$")
pl.add(ellrange,clkk,color='r',label='Input KK')
pl.add(ellrange,clkg,color='g',label='Input KG')
pl.add(ellrange,clgg,color='b',label='Input GG')
y,yerr = gstats("ukk")
print(ells_uncoupled.shape,y.shape,yerr.shape)
pl.add_err(ells_uncoupled,y,yerr=yerr,color='r',label='Uncoupled',marker="o",ls="none")
y,yerr = gstats("ckk")
pl.add_err(ells_coupled+10,y,yerr=yerr,color='r',label='Coupled',alpha=0.6,marker="x",ls="none")
y,yerr = gstats("uke")
pl.add_err(ells_uncoupled,y,yerr=yerr,color='g',marker="o",ls="none")
y,yerr = gstats("cke")
pl.add_err(ells_coupled+10,y,yerr=yerr,color='g',alpha=0.6,marker="x",ls="none")
y,yerr = gstats("uee")
pl.add_err(ells_uncoupled,y,yerr=yerr,color='b',marker="o",ls="none")
y,yerr = gstats("cee")
pl.add_err(ells_coupled+10,y,yerr=yerr,color='b',alpha=0.6,marker="x",ls="none")
pl.legend(loc='upper right')
pl._ax.set_xlim(0,3100)
pl._ax.set_ylim(1.e-11,5e-7)
pl.done(io.dout_dir+field+"_cls.png")


pl = io.Plotter(xlabel="$L$",ylabel="$C_L$")
y,yerr = gstats("ukb")
pl.add_err(ells_uncoupled,y,yerr=yerr,marker="o",ls="none",label="KB")
y,yerr = gstats("ckb")
pl.add_err(ells_coupled,y,yerr=yerr,alpha=0.6)
y,yerr = gstats("ueb")
pl.add_err(ells_uncoupled,y,yerr=yerr,marker="o",ls="none",label="EB")
y,yerr = gstats("ceb")
pl.add_err(ells_coupled,y,yerr=yerr,alpha=0.6)
y,yerr = gstats("ube")
pl.add_err(ells_uncoupled,y,yerr=yerr,marker="o",ls="none",label="BE")
y,yerr = gstats("cbe")
pl.add_err(ells_coupled,y,yerr=yerr,alpha=0.6)
y,yerr = gstats("ubb")
pl.add_err(ells_uncoupled,y,yerr=yerr,marker="o",ls="none",label="BB")
y,yerr = gstats("cbb")
pl.add_err(ells_coupled,y,yerr=yerr,alpha=0.6)
pl.legend(loc='upper right')
pl.hline()
pl._ax.set_xlim(0,3100)
pl._ax.set_ylim(-5.e-10,5e-10)
pl.done(io.dout_dir+field+"_clsnull.png")



clkkfunc = interp1d(ellrange,clkk,bounds_error=False,fill_value=0.)
clkgfunc = interp1d(ellrange,clkg,bounds_error=False,fill_value=0.)
clggfunc = interp1d(ellrange,clgg,bounds_error=False,fill_value=0.)

pl = io.Plotter(xlabel="$L$",ylabel='$\\frac{\\Delta \\sigma(C_L)}{\\sigma(C_L)}$')
y,yerr = gstats("ukk")
yt = binit(clkkfunc(modlmap))
pl.add_err(ells_uncoupled,y/yt-1.,yerr=yerr/yt,color='r',label='kk',marker="o",ls="none")
ym,yerrm = gstats("mtt")
pl.add_err(ells_uncoupled,ym/yt-1.,yerr=yerrm/yt,color='C0',label='tt',marker="x",ls="none",alpha=0.3)
y,yerr = gstats("uke")
yt = binit(clkgfunc(modlmap))
pl.add_err(ells_uncoupled,y/yt-1.,yerr=yerr/yt,color='g',label='kg',marker="o",ls="none")
ym,yerrm = gstats("mte")
pl.add_err(ells_uncoupled,ym/yt-1.,yerr=yerrm/yt,color='C1',label='te',marker="x",ls="none",alpha=0.3)
y,yerr = gstats("uee")
yt = binit(clggfunc(modlmap))
pl.add_err(ells_uncoupled,y/yt-1.,yerr=yerr/yt,color='b',label='gg',marker="o",ls="none")
ym,yerrm = gstats("mee")
pl.add_err(ells_uncoupled,ym/yt-1.,yerr=yerrm/yt,color='C2',label='ee',marker="x",ls="none",alpha=0.3)
pl.legend(loc='upper right')
pl._ax.set_xlim(0,3100)
pl._ax.set_ylim(-0.2,0.15)
pl.hline()
pl.done(io.dout_dir+field+"_clsdiff.png")


LF = cosmology.LensForecast()
LF.loadKK(ellrange,clkk,ellrange,clkk*0.)
LF.loadKS(ellrange,clkg)
LF.loadSS(ellrange,clgg,ngal=2.e6)


ell_edges = bin_edges
ells = ells_uncoupled #(ell_edges[:-1]+ell_edges[1:])/2.
fsky = area/41250.

# Get S/N and errors
sn,errs = LF.sn(ell_edges,fsky,"ks")

print("Expected S/N :",sn)

cerr = s.stats['uke']['err']

pl = io.Plotter(xlabel='$L$',ylabel='$\\frac{\\Delta \\sigma(C_L)}{\\sigma(C_L)}$')
pl.add(ells,(cerr-errs)/errs)
pl.hline()
pl.legend()
pl.done(io.dout_dir+field+"_theoryerrs.png")

