from __future__ import print_function
from orphics import maps,io,cosmology,catalogs
from enlib import enmap,bench
import numpy as np

shape,wcs = maps.rect_geometry(width_deg=20.,px_res_arcmin=0.5)
bigmap = enmap.zeros(shape,wcs)

Nobj = 1000
ras,decs = catalogs.random_catalog(shape,wcs,Nobj,edge_avoid_deg=1.)

arcmin_width = 30.
res = np.min(bigmap.extent()/bigmap.shape[-2:])*180./np.pi*60.
Npix = int(arcmin_width/res)*1.
if Npix%2==0: Npix += 1
cshape,cwcs = enmap.geometry(pos=(0.,0.),res=res/(180./np.pi*60.),shape=(Npix,Npix))
cmodrmap = enmap.modrmap(cshape,cwcs)

sigmas = []
for ra,dec in zip(ras,decs):
    iy,ix = enmap.sky2pix(shape,wcs,(dec*np.pi/180.,ra*np.pi/180.))

    sigma = np.random.normal(3.0,1.0)*np.pi/180./60.
    paste_in = np.exp(-cmodrmap**2./2./sigma**2.)
    bigmap[int(iy-Npix/2):int(iy+Npix/2),int(ix-Npix/2):int(ix+Npix/2)] += paste_in
    sigmas.append(sigma)
    
    
io.plot_img(bigmap,"cat.png",high_res=True)

print("done")

st = maps.Stacker(bigmap,arcmin_width=30.)
stack = 0.
for ra,dec in zip(ras,decs):
    stack += st.cutout(ra*np.pi/180.,dec*np.pi/180.)

io.plot_img(stack)

st = maps.InterpStack(arc_width=30.,px=0.5)
stack = 0.
for ra,dec in zip(ras,decs):
    stack += st.cutout(bigmap,ra,dec)

io.plot_img(stack)


# sys.exit()
# sigma_mean = 3.0 * np.pi/180./60.
# st = maps.ScaleStacker(bigmap,arcmin_width=30.)
# stack = 0.
# for ra,dec,sigma in zip(ras,decs,sigmas):

#     nshape,nwcs = 
#     #stack += st.cutout(ra*np.pi/180.,dec*np.pi/180.,scale=sigma/sigma_mean)

# io.plot_img(stack)






