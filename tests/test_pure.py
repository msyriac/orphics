from enlib import enmap
#import orphics.m as fmaps
import orphics.maps as maps
import orphics.io as io
import orphics.stats as stats
from orphics.cosmology import Cosmology
import numpy as np
import os, sys

out_dir = os.environ['WWW']+"plots/pureTest_"
cc = Cosmology(lmax=6000,pickling=True,dimensionless=False)
theory = cc.theory
deg = 20.
px = 1.0
shape, wcs = maps.rect_geometry(width_deg=deg,px_res_arcmin=px,pol=True)
pa = maps.PatchArray(shape,wcs,cc=cc,orphics_is_dimensionless=False)
ulensed = pa.get_unlensed_cmb()
kappa = pa.get_grf_kappa()
cmb = pa.get_lensed(ulensed,order=5)

# io.highResPlot2d(cmb[0],out_dir+"t.png")
# io.highResPlot2d(cmb[1],out_dir+"q.png")
# io.highResPlot2d(cmb[2],out_dir+"u.png")

modlmap = enmap.modlmap(shape,wcs)
fc = maps.FourierCalc(shape,wcs)
lbin_edges = np.arange(200,6000,40)
lbinner = stats.bin2D(modlmap,lbin_edges)

def plot_powers(cmb,suffix,power=None,w2=1.):

    if power is None:
        power,lteb1,lteb2 = fc.power2d(cmb,pixel_units=False,skip_cross=True)
    power /= w2
    cents,dtt = lbinner.bin(power[0,0])
    cents,dee = lbinner.bin(power[1,1])
    cents,dbb = lbinner.bin(power[2,2])


    pl = io.Plotter(yscale='log')
    ellrange = np.arange(200,6000,1)
    cltt = theory.lCl('TT',ellrange)
    clee = theory.lCl('EE',ellrange)
    clbb = theory.lCl('BB',ellrange)
    pl.add(ellrange,cltt*ellrange**2.,color="k")
    pl.add(ellrange,clee*ellrange**2.,color="k")
    pl.add(ellrange,clbb*ellrange**2.,color="k")
    pl.add(cents,dtt*cents**2.)
    pl.add(cents,dee*cents**2.)
    pl.add(cents,dbb*cents**2.)
    pl.done(out_dir+"powers_"+suffix+".png")


plot_powers(cmb,suffix="periodic",w2=1.)    
taper,w2 = maps.get_taper(shape,taper_percent = 12.0,pad_percent = 3.0,weight=None)
plot_powers(cmb*taper,suffix="tapered",w2=w2)    

print("Pure...")
from orphics import pure
#windict = pure.init_deriv_window(taper,px*np.pi/180./60.)
windict = maps.init_deriv_window(taper,px*np.pi/180./60.)
lxMap,lyMap,modLMap,angLMap,lx,ly = maps.get_ft_attributes(shape,wcs)
#cmb *= taper
#fT, fE, fB =  pure.iqu_to_pure_lteb(cmb[0],cmb[1],cmb[2],modlmap,angLMap,windowDict=windict,method='pure')
fT, fE, fB =  maps.iqu_to_pure_lteb(cmb[0]*taper,cmb[1]*taper,cmb[2]*taper,modlmap,angLMap,windowDict=windict,method='pure')
power = np.zeros((3,3,shape[-2],shape[-1]))
power[0,0,:,:] = fc.f2power(fT,fT)
power[1,1,:,:] = fc.f2power(fE,fE)
power[2,2,:,:] = fc.f2power(fB,fB)
plot_powers(None,suffix="pure",power=power,w2=w2)    
                 

# def mcm(power,bin_edges,\
#         kmask,\
#         transfer = None,\
#         binningWeightMap = None):
#     """
#     window: data window
    
#     """
#     binLo = bin_edges[:-1]
#     binHigh = bin_edges[1:]
    
#     powerMask = power2d(window)
#     powerMaskShifted = fftshift(powerMask)
#     phlx = fftshift(lx)
#     phly = fftshift(ly)
#     if transfer != None:
#         ell, f_ell = transfer
#         t.powerMap *= pixW2d
#         transferTrim = t.trimAtL(trimAtL)
        
        
#     if binningWeightMap ==None:
#         binningWeightMap = powerMask.copy()*0.+1.0
#     else:
#         assert(binningWeightMap.shape == powerMask.shape)
        
        
#     powerMaskTrim = powerMask*kmask
    
#     pMMaps = []
    
#     mArray = numpy.zeros(shape=(len(binLo),len(binLo)))
#     Bbl = numpy.zeros(shape=(len(binLo),lmax)
    
#     modIntLMap = numpy.array(modlmap + 0.5,dtype='int64')
#     cumTerms = 0
#     for ibin in xrange(len(binLo)):
        
#         location = numpy.where((modIntLMap >= binLo[ibin]) & (modIntLMap <= binHi[ibin]))
#         binMap = powerMask.copy()*0.
#         binMap[location] = binningWeightMap[location]
#         sumBin = binMap.sum()
#         binMap[location] *= powerMaskHolder.modLMap[location]**powerOfL
#         assert(sumBin>0.)
        
#         binMap0 = (trimShiftKMap(powerMaskShifted,trimAtL,0,0,\
#                                                phlx,phly))*0.

        
#         t0 = time.time()
#         deltaT = 0.
#         cumTerms += len(location[0])
#         for i in xrange(len(location[0])):
#             ly = powerMaskHolder.ly[location[0][i]]
#             lx = powerMaskHolder.lx[location[1][i]]
#             #  print ly, lx, trimAtL
#             t000 = time.time()
#             binMap0 += binMap[location[0][i],location[1][i]]*\
#                        (trimShiftKMap(powerMaskShifted,trimAtL,lx,ly,\
#                                       phlx,phly))
#             deltaT += (time.time() -t000)
#         binMap0 = ifftshift(binMap0)
#         if transfer != None:
#             pMMaps.append(binMap0[:]*transferTrim.powerMap[:]/sumBin)
#         else:
#             pMMaps.append(binMap0/sumBin)
#               %(ibin, binLo[ibin],binHi[ibin],len(binLo),(time.time() - t0))
        
#     larray = numpy.arange(numpy.int(trimAtL))
#     deltaLx = numpy.abs(powerMaskHolderTrim.modLMap[0,1] - powerMaskHolderTrim.modLMap[0,0])
#     deltaLy = numpy.abs(powerMaskHolderTrim.modLMap[1,0] - powerMaskHolderTrim.modLMap[0,0])
#     delta = numpy.min([deltaLx,deltaLy])/2.0
#     gaussFactor = 1./numpy.sqrt(2.*numpy.pi*delta**2.)

#     modLMapInt = numpy.array(powerMaskHolderTrim.modLMap+0.5, dtype='int64')

#     t0 = time.time()
    
#     for j in xrange(len(binHi)):
#         location = numpy.where((modLMapInt >= binLo[j]) &\
#                                (modLMapInt <= binHi[j]))
#         binMapTrim = powerMaskTrim.copy()*0.
#         binMapTrim[location] = 1.
#         binMapTrim[location] *= numpy.nan_to_num(1./(powerMaskHolderTrim.modLMap[location])**powerOfL)
#         for i in xrange(len(pMMaps)):
#             newMap = pMMaps[i].copy()
#             result = (newMap*binMapTrim).sum()
#             mArray[i,j] = result

#     print "MArray done in %f secs"%(time.time()-t0)
        
#     modMap = numpy.ravel(powerMaskHolderTrim.modLMap)    
#     deltaLx = numpy.abs(powerMaskHolderTrim.lx[1]-powerMaskHolderTrim.lx[0])
#     deltaLy = numpy.abs(powerMaskHolderTrim.ly[1]-powerMaskHolderTrim.ly[0])
    

#     t0 = time.time()
#     for k in xrange(len(larray)):
        
#         gauss = numpy.exp(-(larray-larray[k])**2./(2.*delta**2.))
#         sum = gauss.sum()
#         gauss = numpy.exp(-(modMap-larray[k])**2./(2.*delta**2.))
#         gauss /= sum 
#         binMapTrim = numpy.reshape(gauss,[powerMaskHolderTrim.Ny,powerMaskHolderTrim.Nx])
        
#         # print "time taken in Bbl NO spline %f"%(time.time()-ta)
        
        
        
#         binMapTrim *= powerMaskHolderTrim.modLMap**powerOfL
#         for i in xrange(len(pMMaps)):
#             #newMap = pMMaps[i].copy()
#             result = (pMMaps[i]*binMapTrim).sum()
#             Bbl[i,k] = result

#     Bbl = numpy.dot(scipy.linalg.inv(mArray), Bbl)
    
#     return mArray
