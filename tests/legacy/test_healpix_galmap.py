from enlib import bench,enmap
import orphics.tools.catalogs as cats
import astropy.io.fits as fits
import orphics.tools.io as io
import os,sys
import numpy as np
import healpy as hp

cache_dir = "/gpfs01/astro/workarea/msyriac/data/depot/cache/"

# deep56 = "/gpfs01/astro/workarea/msyriac/data/act/maps/c7v5/s15/deep56/s15_mr2_deep56_pa1_f150_night_set0123_hits.fits"
# d56 = enmap.read_map(deep56)
# with bench.show("to healpix128"):
#     hpmapd56 = d56.to_healpix(nside=128)

# hp.write_map(cache_dir+"d56_nside128_mask.fits",hpmapd56)
# # io.quickMapView(hpmapd56,saveLoc=io.dout_dir+"hpmapd56.png",transform='C')


# fbossn = "/gpfs01/astro/workarea/msyriac/data/act/maps/boss_north/s15/boss_north_2015_150GHz_MR1_PA2_night_set0123_weights_I.fits"
# bossn = enmap.read_map(fbossn)
# with bench.show("to healpix128"):
#     hpmapbossn = bossn.to_healpix(nside=128)

# hp.write_map(cache_dir+"bossn_nside128_mask.fits",hpmapbossn)
# # io.quickMapView(hpmapbossn,saveLoc=io.dout_dir+"hpmapbossn.png",transform='C')


# fs16 = "/gpfs01/astro/workarea/msyriac/data/act/maps/s16/s16_cmb_pa2_f150_night_4way_tot_sky_div_mono_00.fits"
# s16 = enmap.read_map(fs16)
# with bench.show("to healpix128"):
#     hpmaps16 = s16.to_healpix(nside=128)

# hp.write_map(cache_dir+"s16_nside128_mask.fits",hpmaps16)
# # io.quickMapView(hpmaps16,saveLoc=io.dout_dir+"hpmaps16.png",transform='C')


# sys.exit()

hpmapd56 = hp.read_map(cache_dir+"d56_nside128_mask.fits")
hpmapd56[hpmapd56<=2000] = 0
hpmapd56[hpmapd56>2000] = 1
io.quickMapView(hpmapd56,saveLoc=io.dout_dir+"hpmapd56.png",transform='C')
hpmapbossn = hp.read_map(cache_dir+"bossn_nside128_mask.fits")
hpmapbossn[hpmapbossn<=1000] = 0
hpmapbossn[hpmapbossn>1000] = 1
io.quickMapView(hpmapbossn,saveLoc=io.dout_dir+"hpmapbossn.png",transform='C')
hpmaps16 = hp.read_map(cache_dir+"s16_nside128_mask.fits")
hpmaps16[hpmaps16>0.00001] = 1
hpmaps16[hpmaps16<=0.00001] = 0
io.quickMapView(hpmaps16,saveLoc=io.dout_dir+"hpmaps16.png",transform='C')

# sys.exit()

def hcat_from_boss_file(boss_file,nside=128):
    
    hdu = fits.open(boss_file)
    ras = hdu[1].data['RA']
    decs = hdu[1].data['DEC']

    hcat = cats.HealpixCatMapper(nside,ras,decs)
    return hcat

catfiles = ["/gpfs01/astro/workarea/msyriac/data/boss/random0_DR12v5_CMASS_North.fits","/gpfs01/astro/workarea/msyriac/data/boss/random1_DR12v5_CMASS_North.fits","/gpfs01/astro/workarea/msyriac/data/boss/random0_DR12v5_CMASS_South.fits","/gpfs01/astro/workarea/msyriac/data/boss/random1_DR12v5_CMASS_South.fits"]

counts = 0
for catfile in catfiles:
    hcat = hcat_from_boss_file(catfile)
    counts += hcat.counts

counts[counts<1] = 0
counts[counts>0] = 1

io.quickMapView(counts,saveLoc=io.dout_dir+"hpmap.png",transform='C')

countsd56 = counts*hpmapd56
countsbossn = counts*hpmapbossn
countss16 = counts*hpmaps16


countss15 = countsd56+countsbossn

area = lambda x : x.sum()*1./x.size*41250.

print area(countsd56)
print area(countsbossn)
print area(countss15)
print area(countss16)

io.quickMapView(countsd56,saveLoc=io.dout_dir+"hpmapcountsd56.png",transform='C')
io.quickMapView(countsbossn,saveLoc=io.dout_dir+"hpmapcountsbossn.png",transform='C')
io.quickMapView(countss15,saveLoc=io.dout_dir+"hpmapcountss15.png",transform='C')
io.quickMapView(countss16,saveLoc=io.dout_dir+"hpmapcountss16.png",transform='C')


