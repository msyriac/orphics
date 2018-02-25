from __future__ import print_function
from orphics import maps,io,cosmology,stats,mpi
from enlib import enmap
import numpy as np
import os,sys

deg = 20.
px = 2.0
shape,wcs = maps.rect_geometry(width_deg=deg,px_res_arcmin=px)
cc = cosmology.Cosmology(lmax=6000,pickling=True)

ells = np.arange(2,6000,1)
cltt = cc.theory.lCl('TT',ells)
#cltt = cc.theory.gCl('kk',ells)

ps = cltt.reshape((1,1,ells.size))

# from scipy.interpolate import interp1d
modlmap = enmap.modlmap(shape,wcs)
#cltt2d = cc.theory.lCl('TT',modlmap)
cltt2d = enmap.spec2flat(shape,wcs,ps)/(np.prod(shape[-2:])/enmap.area(shape,wcs ))

mg = maps.MapGen(shape,wcs,ps)
fc = maps.FourierCalc(shape,wcs)

bin_edges = np.arange(40,5000,40)
binner = stats.bin2D(modlmap,bin_edges)

#cltt2d = cc.theory.gCl('kk',modlmap)
cents,p1dth = binner.bin(cltt2d)

N = 2000


# MPI
comm = mpi.MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()
Nsims = N
Njobs = Nsims
num_each,each_tasks = mpi.mpi_distribute(Njobs,numcores)
if rank==0: print ("At most ", max(num_each) , " tasks...")
my_tasks = each_tasks[rank]


st = stats.Stats(comm)

import flipper.liteMap as lm
lmap = lm.makeEmptyCEATemplate(deg, deg,meanRa = 180., meanDec = 0.,\
                               pixScaleXarcmin = px, pixScaleYarcmin=px)
    

for i,task in enumerate(my_tasks):
    imap = mg.get_map()
    p2d,_,_ = fc.power2d(imap)
    cents,p1d = binner.bin(p2d)

    st.add_to_stats("p1diff", (p1d.copy()-p1dth)/p1dth)
    
    lmap.fillWithGaussianRandomField(ells,cltt,bufferFactor = 1)
    nmap = enmap.enmap(lmap.data,imap.wcs)
    p2d,_,_ = fc.power2d(nmap)
    cents,p1d = binner.bin(p2d)
    st.add_to_stats("p1diffFlipper", (p1d.copy()-p1dth)/p1dth)

    
    

    if rank==0: print ("Rank 0 done with task ", task+1, " / " , len(my_tasks))
    

st.get_stats()

if rank==0:
    pl = io.Plotter()
    pl.add_err(cents,st.stats["p1diff"]['mean'],yerr=st.stats["p1diff"]['errmean'],marker="o",ls="-",label="enlib")
    pl.add_err(cents,st.stats["p1diffFlipper"]['mean'],yerr=st.stats["p1diffFlipper"]['errmean'],marker="o",ls="-",label="flipper")
    pl.hline()
    pl.legend()
    pl.done(io.dout_dir+"mapgen.png")
