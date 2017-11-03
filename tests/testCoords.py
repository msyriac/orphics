import flipper.liteMap as lm
import btip.inpaintSims as inp
import orphics.tools.io as io
import pyfits
import os, sys
import orphics.analysis.flatMaps as fmaps

out_dir = os.environ['WWW']
lmap = lm.liteMapFromFits("/gpfs01/astro/www/msyriac/forTeva/s15_pa2_s15_pa2_day_s15_pa3_150_fullex_I_map.fits")
Ny, Nx = lmap.data.shape
hdu = pyfits.open("/gpfs01/astro/www/msyriac/forTeva/ACTPol_BOSS-N_BJF.fits")
catalog  = hdu[1].data
ras = hdu[1].data['RADeg']
decs = hdu[1].data['decDeg']
print((catalog.columns))

sys.exit()

print((ras.min()))
print((ras.max()))
#sys.exit()


ras = ras[ras>160]
decs = decs[ras>160]

ras = ras[ras<233]
decs = decs[ras<233]

print((len(ras)))
print((len(decs)))

c = 0
pad = 20
ixs = []
iys = []
for ra,dec in zip(ras,decs):

    ix,iy = lmap.skyToPix(ra,dec)
    #ix,iy = lmap.skyToPix(ra,dec)

    if ix>pad and iy>pad and ix<Nx-pad and iy<Ny-pad and ra>160 and ra<233:
        print((ix,iy))
        c += 1
        ixs.append(ix)
        iys.append(iy)



print(c)

#sys.exit()

holeArc = 10.
holeFrac = 1.

ra_range = [lmap.x1-360.,lmap.x0] 
dec_range = [lmap.y0,lmap.y1]

stack, cents, recons = fmaps.stack_on_map(lmap,60.,0.5,ra_range,dec_range,ras,decs)


io.quickPlot2d(stack,out_dir+"stack.png")

pl = io.Plotter(labelX='Distance from Center (arcminutes)',labelY='Temperature Fluctuation ($\mu K$)', ftsize=10)
pl.add(cents,recons)
pl._ax.axhline(y=0.,ls="--",alpha=0.5)
pl.done(out_dir+"profiles.png")


mask = inp.maskLiteMap(lmap,iys,ixs,holeArc,holeFrac)

io.highResPlot2d(mask.data,out_dir+"mask.png")
