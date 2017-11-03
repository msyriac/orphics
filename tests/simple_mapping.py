from enlib import enmap
import numpy as np
import orphics.tools.io as io
import os,sys

from orphics.tools.catalogs import CatMapper
    
out_dir = os.environ['WWW']+"plots/"

degX = 5.
degY = 20.
px = 1.0
shape, wcs = enmap.rect_geometry(degX*60.,px,height_arcmin=degY*60.)

gal_map = enmap.zeros(shape,wcs)

#y0,x0,y1,x1
bbox = gal_map.box()
print((bbox*180./np.pi))

N = int(1e6)

n_per_pix = N*1./gal_map.size
print((n_per_pix , " galaxies per pixel."))

print("generating random...")
decs = np.random.uniform(bbox[0,0],bbox[1,0],size=N)
ras = np.random.uniform(bbox[0,1],bbox[1,1],size=N)

print((ras.min()*180./np.pi,ras.max()*180./np.pi))
print((decs.min()*180./np.pi,decs.max()*180./np.pi))
# coords = np.vstack((decs,ras))

# print "getting pixels..."
# pixs = gal_map.sky2pix(coords)


# print "binning..."
# dat,xedges,yedges = np.histogram2d(pixs[1,:],pixs[0,:],bins=shape)

mapper = CatMapper(shape,wcs,ras*180./np.pi,decs*180./np.pi)
gal_map = mapper.get_map()

print((gal_map.sum()))
print((gal_map.sum()-N))

gal_map = enmap.smooth_gauss(gal_map,2.0*np.pi/180./60.)

#gal_map = enmap.smooth_gauss(enmap.ndmap(dat.astype(np.float32),wcs),2.0*np.pi/180./60.)
print("plotting...")
io.highResPlot2d(gal_map,out_dir+"galmap.png")


fc = enmap.FourierCalc(shape,wcs)
p2d,d,d = fc.power2d(gal_map)

io.quickPlot2d(np.fft.fftshift(np.log10(p2d)),out_dir+"logp2d.png")
io.quickPlot2d(np.fft.fftshift((p2d)),out_dir+"p2d.png")
