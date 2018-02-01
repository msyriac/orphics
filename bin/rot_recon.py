from orphics.mpi import MPI
import orphics.pipelines as utils
import argparse
from enlib import enmap


# Parse command line
parser = argparse.ArgumentParser(description='Run south rotation test.')
parser.add_argument("-x", "--patch-width", type=float, default=40., help="Patch width in degrees.")
parser.add_argument("-y", "--patch-height", type=float, default=15., help="Patch height in degrees.")
parser.add_argument("-o", "--yoffset", type=float, default=60., help="Offset in declination of southern patch center.")
parser.add_argument("-p", "--full-sky-pixel", type=float, default=0.5,help="Full sky pixel resolution in arcminutes.")
parser.add_argument("-i", "--pix-inter", type=float, default=None,help="Intermediate patch pixelization.")
parser.add_argument("-l", "--lmax", type=int, default=7000,help="Lmax for full-sky lensing.")
parser.add_argument("-b", "--bin-lmax", type=int, default=3000,help="Lmax for binning.")
parser.add_argument("-N", "--Nsims", type=int, default=10,help="Number of sims.")
parser.add_argument("-m", "--meanfield", type=str, default=None,help="Meanfield file root.")
parser.add_argument('-s', "--skip-recon",action='store_true',help="Skip reconstruction.")
args = parser.parse_args()


# Intialize the rotation testing pipeline
pipe = utils.RotTestPipeline(full_sky_pix=args.full_sky_pixel,wdeg=args.patch_width,
                             hdeg=args.patch_height,yoffset=args.yoffset,
                             mpi_comm=MPI.COMM_WORLD,nsims=args.Nsims,lmax=args.lmax,pix_intermediate=args.pix_inter,
                             bin_lmax=args.bin_lmax)

cmb = {} # this will store CMB maps 
ikappa = {} # this will store input kappa maps
mlist = ['e','s','r'] # e stands for patch native to equator, s for native to south, r for rotated from south to equator
mf = {}

# Check if a meanfield is provided
for m in mlist:
    if args.meanfield is not None:
        mf[m] = enmap.read_map(args.meanfield+"/meanfield_"+m+".hdf")
    else:
        mf[m] = 0.

for k,index in enumerate(pipe.tasks):

    # Make CMB maps and kappa maps
    cmb['s'],cmb['e'],ikappa['s'],ikappa['e'] = pipe.make_sim(index)
    # Rotate CMB map and kappa
    cmb['r'] = pipe.rotator.rotate(cmb['s'])
    ikappa['r'] = pipe.rotator.rotate(ikappa['s'], order=5, mode="constant", cval=0.0, prefilter=True, mask_nan=True, safe=True)

    # For each of e,s,r
    for m in mlist:

        # Calculate CMB power
        cxc,kcmb,kcmb = pipe.fc[m].power2d(cmb[m])
        pipe.mpibox.add_to_stats("cmb-"+m,pipe.binner[m].bin(cxc/pipe.w2[m])[1]) # Divide by w2 window correction

        # Calculate input kappa power
        ixi,kinput,_ = pipe.fc[m].power2d(ikappa[m])
        ixi /= pipe.w2[m] # divide by w2 window correction
        pipe.mpibox.add_to_stats("ixi-"+m,pipe.binner[m].bin(ixi)[1])

        
        if args.skip_recon: continue
        if pipe.rank==0: pipe.logger.info( "Reconstructing...")

        # Reconstruct and subtract meanfield if any
        recon = pipe.reconstruct(m,cmb[m]) - mf[m]

        if pipe.rank==0: pipe.logger.info( "Powers...")

        # Calculate raw Clkk power
        rxr,krecon,_ = pipe.fc[m].power2d(recon)
        rxr /= pipe.w4[m]
        # Calculate recon cross input power
        rxi = pipe.fc[m].f2power(kinput,krecon)
        rxi /= pipe.w3[m]
        # Calculate realization dependent N0 ("super dumb")
        n0 = pipe.qest[m].N.super_dumb_N0_TTTT(cxc)/pipe.w2[m]**2.
        # Calculate corrected Clkk power
        rxr_n0 = rxr - n0

        # Collect statistics
        pipe.mpibox.add_to_stack("meanfield-"+m,recon)
        
        pipe.mpibox.add_to_stats("rxr-"+m,pipe.binner[m].bin(rxr)[1])
        pipe.mpibox.add_to_stats("rxi-"+m,pipe.binner[m].bin(rxi)[1])
        pipe.mpibox.add_to_stats("n0-"+m,pipe.binner[m].bin(n0)[1])
        pipe.mpibox.add_to_stats("rxr-n0-"+m,pipe.binner[m].bin(rxr_n0)[1])

    


        if k==0 and pipe.rank==0:
            import orphics.io as io
            io.plot_img(cmb[m],io.dout_dir+"cmb_"+m+".png",high_res=True)
            io.plot_img(recon,io.dout_dir+"recon_"+m+".png",high_res=True)
    
    
    
if pipe.rank==0: pipe.logger.info( "MPI Collecting...")
pipe.mpibox.get_stacks(verbose=False)
pipe.mpibox.get_stats(verbose=False)

if pipe.rank==0:
    pipe.dump(save_meanfield=(args.meanfield is None),skip_recon=args.skip_recon)
