print "Starting imports..."
import matplotlib
matplotlib.use('Agg')
from orphics.analysis.quadraticEstimator import Estimator
import orphics.analysis.flatMaps as fmaps 
from orphics.tools.cmb import loadTheorySpectraFromCAMB
import numpy as np
from astLib import astWCS, astCoords
import liteMap as lm
from orphics.tools.output import Plotter
from orphics.tools.stats import binInAnnuli
import sys

from scipy.interpolate import interp1d
#from scipy.fftpack import fft2,ifft2,fftshift,ifftshift,fftfreq
from scipy.fftpack import fftshift,ifftshift,fftfreq
from pyfftw.interfaces.scipy_fftpack import fft2
from pyfftw.interfaces.scipy_fftpack import ifft2

import pyfftw
pyfftw.interfaces.cache.enable()

import fftTools as ft
import orphics.tools.stats as stats

import fftPol as fpol

from orphics.tools.stats import getStats

from mpi4py import MPI

print "Done with imports..."

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    


loadFile = "/astro/astronfs01/workarea/msyriac/act/normDec14_0.fits" #None
saveFile = None #"/astro/astronfs01/workarea/msyriac/act/normDec14_0.fits"


polCombList = ['TT','EE','EB','TB','TE','ET']
colorList = ['red','blue','green','orange','purple','brown']

simRoot = "/astro/astronfs01/workarea/msyriac/cmbSims/"

lensedTPath = lambda x: simRoot + "lensedCMBMaps_" + str(x).zfill(5) + "/order5_periodic_lensedCMB_T_beam_0.fits"
lensedQPath = lambda x: simRoot + "lensedCMBMaps_" + str(x).zfill(5) + "/order5_periodic_lensedCMB_Q_beam_0.fits"
lensedUPath = lambda x: simRoot + "lensedCMBMaps_" + str(x).zfill(5) + "/order5_periodic_lensedCMB_U_beam_0.fits"
kappaPath = lambda x: simRoot + "phiMaps_" + str(x).zfill(5) + "/kappaMap_0.fits"
beamPath = simRoot + "beam_0.txt"

l,beamells = np.loadtxt(beamPath,unpack=True,usecols=[0,1])

cmbellmin = 100
cmbellmax = 3000
kellmin = 100
kellmax = 3000

#cambRoot = "/astro/u/msyriac/repos/cmb-lensing-projections/data/TheorySpectra/ell28k_highacc"
cambRoot = "/astro/u/msyriac/repos/actpLens/data/non-linear"

TCMB = 2.7255e6
theory = loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=False,useTotal=False,TCMB = TCMB,lpad=4000)

ellkk = np.arange(2,9000,1)
Clkk = theory.gCl("kk",ellkk)    

N = int(sys.argv[1])

assert N%numcores==0

num_each = (N / numcores)
startIndex = rank*num_each
endIndex = startIndex + num_each
myIs = range(N)[startIndex:endIndex]

    

listCrossPower = {}
listReconPower = {}



for polComb in polCombList:
    listCrossPower[polComb] = []
    listReconPower[polComb] = []




bin_edges = np.arange(kellmin,kellmax,150)

for k,i in enumerate(myIs):
    print i

    lensedTLm = lm.liteMapFromFits(lensedTPath(i))
    lensedQLm = lm.liteMapFromFits(lensedQPath(i))
    lensedULm = lm.liteMapFromFits(lensedUPath(i))
    kappaLm = lm.liteMapFromFits(kappaPath(i))

    

    if k==0:
        lxMap,lyMap,modLMap,thetaMap,lx,ly = fmaps.getFTAttributesFromLiteMap(lensedTLm)
        beamTemplate = fmaps.makeTemplate(l,beamells,modLMap)
        fMaskCMB = fmaps.fourierMask(lx,ly,modLMap,lmin=cmbellmin,lmax=cmbellmax)
        fMask = fmaps.fourierMask(lx,ly,modLMap,lmin=kellmin,lmax=kellmax)




    fot,foe,fob = fmaps.TQUtoFourierTEB(lensedTLm.data.copy().astype(float)/TCMB,lensedQLm.data.copy().astype(float)/TCMB,lensedULm.data.copy().astype(float)/TCMB,modLMap,thetaMap)

        

    fot[:,:] = (fot[:,:] / beamTemplate[:,:])
    foe[:,:] = (foe[:,:] / beamTemplate[:,:])
    fob[:,:] = (fob[:,:] / beamTemplate[:,:])
    noise = fot.copy()*0.
    

    if k==0:


        qest = Estimator(lensedTLm,
                         theory,
                         theorySpectraForNorm=None,
                         noiseX2dTEB=[noise,noise,noise],
                         noiseY2dTEB=[noise,noise,noise],
                         fmaskX2dTEB=[fMaskCMB]*3,
                         fmaskY2dTEB=[fMaskCMB]*3,
                         fmaskKappa=fMask,
                         doCurl=False,
                         TOnly=False,
                         halo=True,
                         gradCut=10000,verbose=True,
                         loadPickledNormAndFilters=loadFile,
                         savePickledNormAndFilters=saveFile)



    print "Reconstructing" , i , " ..."
    qest.updateTEB_X(fot,foe,fob,alreadyFTed=True)
    qest.updateTEB_Y(alreadyFTed=True)

    for j, polComb in enumerate(polCombList):

        kappa = qest.getKappa(polComb)


        reconLm = lensedTLm.copy()
        reconLm.data[:,:] = kappa[:,:].real

        print "crossing with input"


        p2d = ft.powerFromLiteMap(kappaLm,reconLm,applySlepianTaper=False)
        centers, means = stats.binInAnnuli(p2d.powerMap, p2d.modLMap, bin_edges)
        listCrossPower[polComb].append( means )



        p2d = ft.powerFromLiteMap(reconLm,applySlepianTaper=False)
        centers, means = stats.binInAnnuli(p2d.powerMap, p2d.modLMap, bin_edges)
        listReconPower[polComb].append( means )

    p2d = ft.powerFromLiteMap(kappaLm,applySlepianTaper=False)
    centers, means = stats.binInAnnuli(p2d.powerMap, p2d.modLMap, bin_edges)

    if k==0: totInputPower = (means.copy()*0.).astype(dtype=np.float64)

    totInputPower = totInputPower + means




if rank!=0:
    for i,polComb in enumerate(polCombList):
        data = np.array(listCrossPower[polComb],dtype=np.float64)
        comm.Send(data.copy(), dest=0, tag=i)
        data = np.array(listReconPower[polComb],dtype=np.float64)
        comm.Send(data.copy(), dest=0, tag=i+80)
        
    comm.Send(totInputPower.copy(), dest=0, tag=800)
        
else:

    totAllInputPower = totInputPower
    rcvTotInputPower = totAllInputPower.copy()*0.


    listAllCrossPower = {}
    listAllReconPower = {}

    for polComb in polCombList:
        listAllCrossPower[polComb] = np.array(listCrossPower[polComb],dtype=np.float64)
        listAllReconPower[polComb] = np.array(listReconPower[polComb],dtype=np.float64)
    

    rcvInputPowerMat = listAllReconPower['TT'].copy()*0.



    for job in range(1,numcores):
        comm.Recv(rcvTotInputPower, source=job, tag=800)
        totAllInputPower = totAllInputPower + rcvTotInputPower

        for i,polComb in enumerate(polCombList):
            comm.Recv(rcvInputPowerMat, source=job, tag=i)
            listAllCrossPower[polComb] = np.vstack((listAllCrossPower[polComb],rcvInputPowerMat))
            comm.Recv(rcvInputPowerMat, source=job, tag=i+80)
            listAllReconPower[polComb] = np.vstack((listAllReconPower[polComb],rcvInputPowerMat))
        

    statsCross = {}
    statsRecon = {}

    pl = Plotter(scaleY='log')
    pl.add(ellkk,Clkk,color='black',lw=2)
    

    for polComb,col in zip(polCombList,colorList):
        statsCross[polComb] = getStats(listAllCrossPower[polComb])
        pl.addErr(centers,statsCross[polComb]['mean'],yerr=statsCross[polComb]['errmean'],ls="none",marker="o",markersize=8,label="recon x input "+polComb,color=col,mew=2,elinewidth=2)

        statsRecon[polComb] = getStats(listAllReconPower[polComb])
        fp = interp1d(centers,statsRecon[polComb]['mean'],fill_value='extrapolate')
        pl.add(ellkk,(fp(ellkk))-Clkk,color=col,lw=2)


    avgInputPower  = totAllInputPower/N
    pl.add(centers,avgInputPower,color='cyan',lw=3) # ,label = "input x input"


    pl.legendOn(labsize=10,loc='lower left')
    pl._ax.set_xlim(kellmin,kellmax)
    pl.done("tests/output/power.png")


    pl = Plotter()

    for polComb,col in zip(polCombList,colorList):
        cross = statsCross[polComb]['mean']
        
        
        pl.add(centers,(cross-avgInputPower)*100./avgInputPower,label=polComb,color=col,lw=2)


    pl.legendOn(labsize=10,loc='upper right')
    pl._ax.set_xlim(kellmin,kellmax)
    pl.done("tests/output/percent.png")


    pl = Plotter()

    for polComb,col in zip(polCombList,colorList):
        cross = statsCross[polComb]['mean']
        crossErr = statsCross[polComb]['errmean']
        recon = statsRecon[polComb]['mean']
        
        pl.add(centers,(cross-avgInputPower)/crossErr,label=polComb,color=col)


    pl.legendOn(labsize=10,loc='upper right')
    pl._ax.set_xlim(kellmin,kellmax)
    pl.done("tests/output/bias.png")




