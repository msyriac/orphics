from __future__ import print_function
from orphics import maps,io,cosmology
from enlib import enmap
import numpy as np
import os,sys


class ACTMaps(object):

    def __init__(self):
        pass

"""
Define a region by a bounding box.
Define 


Tcutout = Tmap(allsplits,array)
Tarray = Tcutout*taper
kTarray = fft(Tarray)
cinv = inv(cov(n2ds,kbeams,fgs))
coadd T = ifft(silc(kTarrays/kbeam,cinv)*kbeam_effective)
effective 2D noise power = silc_noise

# REQUIRED INPUTS
- coadd T,E,B maps, simulated coadd map
- effective beam (requires choice)
- effective 2D noise power (requires splits)
Lensing map = QE(coaddX,coaddY,beam_effectiveX,noise_effectiveX,beam_effectiveY,noise_effectiveY)
A lensing map requires an optimal coadd of all the data in a region.
We want to end up with a lensing map and simulations of the lensing map.


"""


