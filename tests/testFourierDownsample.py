import os
import numpy as np
import orphics.analysis.flatMaps as fmaps
import orphics.tools.io as io
import flipper.liteMap as lm
from enlib.fft import fft,ifft
from numpy.fft import fftshift,ifftshift
from enlib.resample import resample_fft,resample_bin

arcX = 20*60.
arcY = 10*60.
arc = 10.*60.
px = 0.5
pxDn = 7.5

fineTemplate = lm.makeEmptyCEATemplate(arcX/60.,arcY/60.,pixScaleXarcmin=px,pixScaleYarcmin=px)
lxMap,lyMap,modLMap,thetaMap,lx,ly = fmaps.getFTAttributesFromLiteMap(fineTemplate)

xMap,yMap,modRMap,xx,yy = fmaps.getRealAttributes(fineTemplate)

# sigArc = 3.0
# sig = sigArc*np.pi/180./60.
# fineTemplate.data = np.exp(-modRMap**2./2./sig**2.)

import btip.inpaintStamp as inp
ell,Cl = np.loadtxt("../btip/data/cltt_lensed_Feb18.txt",unpack=True)
ell,Cl = inp.total_1d_power(ell,Cl,ellmax=modLMap.max(),beamArcmin=1.4,noiseMukArcmin=12.0,TCMB=2.7255e6)
twoDPower = inp.spec1dTo2d(fineTemplate,ell,Cl)#,k=1) # !!!!!
fineTemplate.data = twoDPower


pl = io.Plotter()
pl.plot2d(np.log10(np.fft.fftshift(fineTemplate.data)))
pl.done(os.environ['WWW']+"ftest.png")



coarseTemplate = lm.makeEmptyCEATemplate(arc/60.,arc/60.,pixScaleXarcmin=pxDn,pixScaleYarcmin=pxDn)
lxMapDn,lyMapDn,modLMapDn,thetaMapDn,lxDn,lyDn = fmaps.getFTAttributesFromLiteMap(coarseTemplate)




# ft = fftshift(fft(fineTemplate.data,axes=[-2,-1]))
print((fineTemplate.data.shape))
# print ft.shape
print((coarseTemplate.data.shape))


Ncy,Ncx = coarseTemplate.data.shape
Nfy,Nfx = fineTemplate.data.shape

# startY = int(Nfy/2.-Ncy/2.)
# endY = int(Nfy/2.+Ncy/2.)

# startX = int(Nfx/2.-Ncx/2.)
# endX = int(Nfx/2.+Ncx/2.)

# ftTrim = ft[startY:endY,startX:endX]

# print ftTrim.shape

#coarseNew = ifft(ifftshift(ftTrim),axes=[-2,-1],normalize=True).real
#np.fft.irfft(np.fft.rfft[:n/2+1],n)


print((Ncy,Ncx))
print((Nfy,Nfx))

coarseNew = resample_bin(fineTemplate.data,factors=[float(Ncx)/Nfx,float(Ncy)/Nfy])
# print coarseNew.shape
# print np.any(coarseNew<0.)
# print np.any(np.isnan(coarseNew))

pl = io.Plotter()
pl.plot2d(np.log10(np.fft.fftshift(coarseNew)))
pl.done(os.environ['WWW']+"coarse.png")
