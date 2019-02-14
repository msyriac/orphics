from __future__ import print_function
from orphics import maps,io,cosmology
from enlib import enmap,fft
import numpy as np
import os,sys
from flipper import liteMap as lm
from flipper import fftTools as ft


shape,wcs = maps.rect_geometry(width_deg=30./60.,px_res_arcmin=0.5)


modrmap = enmap.modrmap(shape,wcs)
modlmap = enmap.modlmap(shape,wcs)

sigma = 2.*np.pi/180./60.
ptsrc = np.exp(-modrmap**2./2./sigma**2.)

io.plot_img(ptsrc)

lmap = enmap.to_flipper(ptsrc)
kmap = ft.fftFromLiteMap(lmap)
modlmap2 = kmap.modLMap

imap = fft.ifft(kmap.kMap,axes=[-2,-1],normalize=True).real  

diff = (modlmap-modlmap2)*100./modlmap
io.plot_img(diff)
print(diff)

