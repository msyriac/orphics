"""
Inpainting holes in rectangular pixel maps.
"""


from orphics import maps,io,cosmology,stats,pixcov,catalogs
from pixell import enmap,utils as u,curvedsky as cs,pointsrcs
import numpy as np
import os,sys


N = 3 # number of sources in map
fwhm = 1.4 # FWHM arcmin
noise = 10.0 # uK-arcmin
lmax = 6000 # signal ellmax
beam_fn = lambda x: maps.gauss_beam(x,fwhm)


# Random sources

np.random.seed(10)

cosdecs = np.random.uniform(-1,1,N)
srcs = np.zeros((N,3))
srcs[:,0] = np.arccos(cosdecs)-np.pi/2.
srcs[:,1] = np.random.uniform(0,2.*np.pi,N)
srcs[:,2] = 1000 # uK peak signal
coords = srcs[:,:2]

# Simulate sources
smap = pointsrcs.sim_srcs(shape[-2:], wcs, srcs, maps.sigma_from_fwhm(fwhm*u.arcmin))

# Q/U sources with 1% flux
smap = np.repeat(smap[None,...],3,0)
smap[1:] = smap[1:] / 100

# 2 arcmin full sky geometry
shape,wcs = enmap.fullsky_geometry(res=2.0 * u.arcmin)
shape = (3,) + shape

# Noise maps
nmap = maps.white_noise(shape,wcs,noise)
nmap[1:,...] = nmap[1:,...] * np.sqrt(2.)


# Observed map
imap = cs.filter(maps.rand_cmb_sim(shape,wcs,lmax),beam_fn,lmax=lmax) + smap + nmap
# Inverse variance map
ivar = maps.ivar(shape[-2:],wcs,noise)


# Inpaint hole radius
hole_radius = 6.0 * u.arcmin
# Output directory to save geometry files
output_dir = './'

# Theory function for signal spectra (must accept e.g. 'TT' as first argument
# and numpy array ells as second argument)
theory = cosmology.default_theory().lCl

# Slow step: make and save geometries. This function can be parallelized over
# the number of sources by passing an MPI communicator in the comm argument.
# This step does not need to be repeated on sims that use the same ivar map!
# Do it once for both data and sims and the result will be saved in the output_dir
# directory.
pixcov.inpaint_uncorrelated_save_geometries(coords,hole_radius,ivar,output_dir,theory_fn=theory,beam_fn=beam_fn)

# Fast step to inpaint imap with geometries from output_dir
omap = pixcov.inpaint_uncorrelated_from_saved_geometries(imap,output_dir)

# Plot source cutouts before inpainting
context_fraction = 2./3.
radius = hole_radius * (1 + context_fraction)
for cutout in pixcov.extract_cutouts(imap,coords,radius):
    io.plot_img(cutout[0])

# Plot source cutouts after inpainting
for cutout in pixcov.extract_cutouts(omap,coords,radius):
    io.plot_img(cutout[0])

