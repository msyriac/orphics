import os,sys
import numpy as np
import orphics.analysis.flatMaps as fmaps
import orphics.tools.io as io
import orphics.tools.stats as stats
from numpy.fft import fftshift
from orphics.theory.cosmology import Cosmology
import orphics.tools.cmb as cmb


out_dir = os.environ['WWW']

class template:
    pass


Ny = 1000
Nx = 1000
px_arc = 0.5

px_rad = px_arc * np.pi/180./60.

pixScaleX,pixScaleY = px_rad,px_rad

size_deg_y = Ny*px_arc/60.
size_deg_x = Nx*px_arc/60.
area = size_deg_y*size_deg_x
print(("Size deg. " , size_deg_y,"x",size_deg_x))
print(("Area deg.sq. " , area))

templateLM = template()
templateLM.Ny, templateLM.Nx = Ny,Nx
templateLM.pixScaleY, templateLM.pixScaleX = px_rad,px_rad


lxMap,lyMap,modLMap,thetaMap,lx,ly = fmaps.getFTAttributesFromLiteMap(templateLM)

#pwindow = fmaps.pixel_window_function(modLMap,thetaMap,pixScaleX,pixScaleY)
pwindow = fmaps.pixwin(modLMap,px_arc)

io.highResPlot2d(fftshift(pwindow),out_dir+"pwindow.png")


ellmin_cmb = 100
ellmax_cmb = 10000
dell_cmb = 100
lmax = 8000
beamArcmin = 1.4
noiseMukArcmin = 16.0
lknee = 3000
alpha = -4.6

bin_edges = np.arange(ellmin_cmb,ellmax_cmb,dell_cmb)
ellfine = np.arange(ellmin_cmb,ellmax_cmb,1)
binner = stats.bin2D(modLMap,bin_edges)

cents, pcls = binner.bin(pwindow)

cc = Cosmology(lmax=lmax,pickling=True)
theory = cc.theory
cltt = theory.lCl('TT',ellfine)

cltt2d = theory.lCl('TT',modLMap)


nfunc = cmb.get_noise_func(beamArcmin,noiseMukArcmin,ellmin=ellmin_cmb,ellmax=ellmax_cmb,TCMB=2.7255e6,lknee=lknee,alpha=alpha)

nells = nfunc(ellfine)

n2d = nfunc(modLMap)

n2dp = n2d/pwindow**2.
cents, nclsp = binner.bin(n2dp)
cents, ncls = binner.bin(n2d)


pl = io.Plotter(scaleY='log',scaleX='log')
pl.add(ellfine,cltt*ellfine**2.)
pl.add(ellfine,nells*ellfine**2.,ls="--")
pl.add(cents,nclsp*cents**2.,ls="-.")
pl.legendOn()
pl.done(out_dir+"pcls.png")


pl = io.Plotter(labelY="% diff of $N_{\ell}$ with pixel",labelX="$\ell$")
pl.add(cents,(nclsp-ncls)*100./ncls)
pl.done(out_dir+"pdiff.png")


