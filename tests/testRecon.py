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


def TQUtoFourierTEB(T_map,Q_map,U_map,modLMap,angLMap):

    fT=fft2(T_map)    
    fQ=fft2(Q_map)        
    fU=fft2(U_map)
    
    fE=fT.copy()
    fB=fT.copy()
    fE[:]=fQ[:]*np.cos(2.*angLMap)+fU*np.sin(2.*angLMap)
    fB[:]=-fQ[:]*np.sin(2.*angLMap)+fU*np.cos(2.*angLMap)
    
    return(fT, fE, fB)
    



polCombList = ['TT','EE','EB','TB','TE','ET']
colorList = ['red','blue','green','orange','purple','brown']

simRoot = "/astro/astronfs01/workarea/msyriac/cmbSims/"

lensedTPath = lambda x: simRoot + "lensedCMBMaps_" + str(x).zfill(5) + "/order5_periodic_lensedCMB_T_beam_0.fits"
lensedQPath = lambda x: simRoot + "lensedCMBMaps_" + str(x).zfill(5) + "/order5_periodic_lensedCMB_Q_beam_0.fits"
lensedUPath = lambda x: simRoot + "lensedCMBMaps_" + str(x).zfill(5) + "/order5_periodic_lensedCMB_U_beam_0.fits"
kappaPath = lambda x: simRoot + "phiMaps_" + str(x).zfill(5) + "/kappaMap_0.fits"
beamPath = simRoot + "beam_0.txt"

l,beamells = np.loadtxt(beamPath,unpack=True,usecols=[0,1])
pl = Plotter()
pl.add(l,beamells)
pl._ax.set_xlim(0.,max(l))
pl.done("tests/output/beam.png")

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

N = 10

avg = {}
avg2 = {}


for polComb in polCombList:
    avg[polComb] = 0.
    avg2[polComb] = 0.



avg3 = 0.

bin_edges = np.arange(kellmin,kellmax,150)


for i in range(N):
    print i

    lensedTLm = lm.liteMapFromFits(lensedTPath(i))
    lensedQLm = lm.liteMapFromFits(lensedQPath(i))
    lensedULm = lm.liteMapFromFits(lensedUPath(i))
    kappaLm = lm.liteMapFromFits(kappaPath(i))

    if i==0:
        lxMap,lyMap,modLMap,thetaMap,lx,ly = fmaps.getFTAttributesFromLiteMap(lensedTLm)
        beamTemplate = fmaps.makeTemplate(l,beamells,modLMap)
        fMaskCMB = fmaps.fourierMask(lx,ly,modLMap,lmin=cmbellmin,lmax=cmbellmax)
        fMask = fmaps.fourierMask(lx,ly,modLMap,lmin=kellmin,lmax=kellmax)




    fot,foe,fob = TQUtoFourierTEB(lensedTLm.data.copy().astype(float)/TCMB,lensedQLm.data.copy().astype(float)/TCMB,lensedULm.data.copy().astype(float)/TCMB,modLMap,thetaMap)

        

    fot[:,:] = (fot[:,:] / beamTemplate[:,:])
    foe[:,:] = (foe[:,:] / beamTemplate[:,:])
    fob[:,:] = (fob[:,:] / beamTemplate[:,:])
    noise = fot.copy()*0.
    

    if i==0:
        # pl = Plotter()
        # pl.plot2d(lensedTLm.data)
        # pl.done("tests/output/withBeam.png")
        # pl = Plotter()
        # pl.plot2d(ifft2(TData*fMaskCMB).real)
        # pl.done("tests/output/withoutBeam.png")


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
                         gradCut=10000,verbose=True)


    print "Reconstructing" , i , " ..."
    #qest.updateTEB_X(TData,alreadyFTed=True)
    #qest.updateTEB_Y(alreadyFTed=True)
    qest.updateTEB_X(fot,foe,fob,alreadyFTed=True)
    qest.updateTEB_Y(alreadyFTed=True)

    pl = Plotter(scaleY='log')
    pl.add(ellkk,Clkk,color='black',lw=2)
    for polComb,col in zip(polCombList,colorList):

        kappa = qest.getKappa(polComb)

        # if i==0:
        #     pl = Plotter()
        #     pl.plot2d(kappa.real)
        #     pl.done("tests/output/kappa.png")


        reconLm = lensedTLm.copy()
        reconLm.data[:,:] = kappa[:,:]

        print "crossing with input"


        p2d = ft.powerFromLiteMap(kappaLm,reconLm,applySlepianTaper=False)
        centers, means = stats.binInAnnuli(p2d.powerMap, p2d.modLMap, bin_edges)
        avg[polComb] = avg[polComb] + means
        plotAvg = avg[polComb].copy()
        plotAvg[plotAvg<=0.] = np.nan


        try:
            pl.add(centers,plotAvg/(i+1),ls="none",marker="o",markersize=8,label="recon x input "+polComb,color=col)
        except:
            pass


        # print np.nanmean(avg[polComb]/(i+1))


        p2d = ft.powerFromLiteMap(reconLm,applySlepianTaper=False)
        centers, means = stats.binInAnnuli(p2d.powerMap, p2d.modLMap, bin_edges)
        avg2[polComb] = avg2[polComb] + means
        plotAvg = avg2[polComb].copy()
        plotAvg[plotAvg<=0.] = np.nan
        fp = interp1d(centers,plotAvg,fill_value='extrapolate')
        pl.add(ellkk,(fp(ellkk)/(i+1))-Clkk,color=col,lw=2) # ,label="recon x recon - clkk "+polComb


        # centers, Nlbinned = binInAnnuli(qest.N.Nlkk, modLMap, bin_edges)
        # pl.add(centers,Nlbinned,ls="--",lw=2,color="orange")


    p2d = ft.powerFromLiteMap(kappaLm,applySlepianTaper=False)
    centers, means = stats.binInAnnuli(p2d.powerMap, p2d.modLMap, bin_edges)
    avg3 = avg3 + means
    plotAvg = avg3.copy()
    plotAvg[plotAvg<=0.] = np.nan
    pl.add(centers,plotAvg/(i+1),color='cyan',lw=3) # ,label = "input x input"

    # print np.nanmean(avg3/(i+1.))




    pl.legendOn(labsize=10,loc='lower left')
    pl._ax.set_xlim(kellmin,kellmax)
    pl.done("tests/output/power.png")



