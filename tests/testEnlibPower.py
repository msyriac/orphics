from enlib import enmap
from szar.counts import ClusterCosmology
from orphics.analysis import flatMaps as fmaps
import orphics.tools.io as io
import orphics.tools.cmb as cmb
import orphics.tools.stats as stats
import os, sys
import numpy as np

out_dir = ""

lmax = 2000

cc = ClusterCosmology(lmax=lmax,pickling=True)
TCMB = 2.7255e6
theory = cc.theory

ps = cmb.enmap_power_from_orphics_theory(theory,lmax,lensed=False)
cov = ps

width_deg = 20.
pix = 2.0
shape, wcs = enmap.get_enmap_patch(width_deg*60.,pix,proj="CAR",pol=True)

m1 = enmap.rand_map(shape, wcs, cov, scalar=True, seed=None, power2d=False,pixel_units=False)
modlmap = m1.modlmap()


io.quickPlot2d(m1[0],out_dir+"m1.png")
cltt = ps[0,0]
ells = np.arange(0,cltt.size)
pl = io.Plotter(scaleY='log')
pl.add(ells,cltt*ells**2.)

debug_edges = np.arange(200,2000,80)
dbinner = stats.bin2D(modlmap,debug_edges)
powermap = fmaps.get_simple_power_enmap(m1[0])
cents, dcltt = dbinner.bin(powermap)

pl.add(cents,dcltt*cents**2.,label="power of map")

pa = fmaps.PatchArray(shape,wcs,skip_real=True)
pa.add_noise_2d(powermap)
nT = pa.nT
cents, dcltt = dbinner.bin(nT)
pl.add(cents,dcltt*cents**2.,label="binned 2d power from pa")

cov = np.zeros((3,3,modlmap.shape[0],modlmap.shape[1]))
cov[0,0] = nT
cov[1,1] = nT
cov[2,2] = nT
m2 = enmap.rand_map(shape, wcs, cov, scalar=True, seed=None, power2d=True,pixel_units=False)
powermap = fmaps.get_simple_power_enmap(m2[0])
cents, dcltt = dbinner.bin(powermap)
pl.add(cents,dcltt*cents**2.,label="power of map made from p2d")

pl.legendOn(labsize=10,loc="lower left")
pl.done("cls.png")
