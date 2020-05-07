'''
Utilities for dealing with galaxy catalogs, projecting catalogs into pixelated maps, etc.
'''


import numpy as np
from pixell import enmap
import healpy as hp
from astropy.io import fits
from orphics import maps

class Pow2Cat(object):
    def __init__(self,ells,clgg,clkg=None,clkk=None,depth_map=None,lmax=None):
        """Initialize a catalog generator

        Args:
            ells: (nells,) array specifying multipoles at which clgg,clkg,clkk are defined
            clgg: (nells,) array containing power spectrum of the field you want to Poisson sample from
            clkg: (nells,) array containing cross-spectrum with the optional field you don't want to Poisson sample from (optional)
            clkk: (nells,) array containing auto-spectrum of the field you don't want to Poisson sample from (optional)
            depth_map: ndmap specifying depth. Max of this array will be divided out.
                       
        """
        ls = np.arange(0,ells.max(),1)
        self.lmax = ls.max()
        clgg = maps.interp(ells,clgg)(ls)
        if clkg is not None:
            assert clkk is not None
            ncomp = 2
            clkg = maps.interp(ells,clkg)(ls)
            clkk = maps.interp(ells,clkk)(ls)
        else:
            ncomp = 1
        self.shape = (ncomp,)+depth_map.shape[-2:]
        self.wcs = depth_map.wcs
        ps = np.zeros((ncomp,ncomp,ls.size))
        ps[0,0] = clgg
        if clkg is not None:
            ps[1,1] = clkk
            ps[0,1] = clkg
            ps[1,0] = clkg
        self.depth_map = depth_map/depth_map.max()
        assert np.all(self.depth_map>=0)
        self.ps = ps
        self.ncomp = ncomp
        
    def get_map(self,seed=None):
        """Get correlated galaxy and kappa map """
        from pixell import curvedsky
        alms = curvedsky.rand_alm_healpy(self.ps, lmax=int(self.lmax)+1, seed=seed)
        ncomp = 1 if len(self.shape)==2 else self.shape[0]
        omap   = enmap.empty((ncomp,)+self.shape[-2:], self.wcs, dtype=np.float64)
        omap = curvedsky.alm2map(alms, omap, spin=0)
        return alms,omap

    def get_cat(self,ngals,seed=None,depth_threshold=0.5,cull_voids=True,add_jitter=True):
        """Get a catalog with total number of galaxies ngals and a kappa map that are correlated."""
        alms,retmap = self.get_map(seed=seed)
        if self.ncomp==1:
            gmap = retmap[0].copy()
        else:
            gmap,kmap = retmap.copy()
            kmap -= kmap.mean()
        gmap -= gmap.mean()
        if cull_voids:
            gmap[gmap<-1] = -1
        else:
            assert gmap.min()>-1, "The galaxy field has too much power and thus regions of underdensity < -1."
        
        gmodmap = gmap.copy()
        dmap = self.depth_map
        dmap[dmap<depth_threshold] = 0
        pdecs,pras = gmap.posmap()
        ngalmap = (gmodmap+1.)*dmap*np.cos(pdecs)
        ngalmap *= (ngals/ngalmap.sum())
        assert np.all(np.isfinite(ngalmap))
        assert np.all(ngalmap>=0)
        sampled = np.random.poisson(ngalmap).astype(np.float64)
        Ny,Nx = self.shape[-2:]
        pixmap = (enmap.pixmap(self.shape,self.wcs)).reshape(2,Ny*Nx)
        nobjs = sampled.reshape(-1).astype(np.int)
        cat = np.repeat(pixmap,nobjs,-1).astype(np.float64)
        jitter = np.random.uniform(-0.5,0.5,size=cat.shape) if add_jitter else 0.
        cat += jitter
        decs,ras = np.rad2deg(enmap.pix2sky(self.shape,self.wcs,cat))
        return ras,decs,alms,retmap

def load_fits(fits_file,column_names,hdu_num=1,Nmax=None):
    hdu = fits.open(fits_file)
    columns = {}
    for col in column_names:
        columns[col] = hdu[hdu_num].data[col][:Nmax]
    hdu.close()
    return columns

def dndz(z,z0=1./3.):
    """A simple 1-parameter dndz parameterization.
    """
    ans = (z**2.)* np.exp(-1.0*z/z0)/ (2.*z0**3.)
    return ans    


def select_region(ra_col,dec_col,other_cols,ra_min,ra_max,dec_min,dec_max):
    """Given ra,decs in ra_col,dec_col and a list of other lists with the
    same size as ra_col and dec_col, return newly selected ras,decs + other
    columns bounded by specified minimum and maximum ra and dec.

    Wraps around 180d.
    """
    from astropy.coordinates import Angle
    import astropy.units as u

    ra_col = Angle(ra_col * u.deg).wrap_at('180d').degree
    ret_cols = []
    for other_col in other_cols:
        ret_cols.append(other_col[np.logical_and(np.logical_and(np.logical_and(ra_col>ra_min,ra_col<ra_max),dec_col>dec_min),dec_col<dec_max)])

    ra_ret = ra_col[np.logical_and(np.logical_and(np.logical_and(ra_col>ra_min,ra_col<ra_max),dec_col>dec_min),dec_col<dec_max)]
    dec_ret = dec_col[np.logical_and(np.logical_and(np.logical_and(ra_col>ra_min,ra_col<ra_max),dec_col>dec_min),dec_col<dec_max)]
    return ra_ret,dec_ret,ret_cols


def random_catalog(shape,wcs,N,edge_avoid_deg=0.):

    box = enmap.box(shape,wcs)
    dec0 = min(box[0,0],box[1,0]) + edge_avoid_deg*np.pi/180.
    dec1 = max(box[0,0],box[1,0]) - edge_avoid_deg*np.pi/180.
    ra0 = min(box[0,1],box[1,1]) + edge_avoid_deg*np.pi/180.
    ra1 = max(box[0,1],box[1,1]) - edge_avoid_deg*np.pi/180.

    ras = np.random.uniform(ra0,ra1,N) * 180./np.pi
    decs = np.random.uniform(dec0,dec1,N) * 180./np.pi

    return ras,decs


class CatMapper(object):
    """Base class for a number of interfaces with galaxy catalogs. Given a geometry
    (either in enlib shape,wcs form or as healpix nside), converts the contents
    of the catalog to pixel positions and bins it into pixelated maps.

    Currently

    """

    def __init__(self,ras_deg=None,decs_deg=None,shape=None,wcs=None,nside=None,verbose=True,hp_coords="equatorial",mask=None,weights=None,pixs=None):

        self.verbose = verbose
        if nside is not None:
            self.nside = nside
            self.shape = hp.nside2npix(nside)
            self.curved = True
        else:
            self.shape = shape
            self.wcs = wcs
            self.curved = False

        if pixs is None:
            if nside is not None:

                eq_coords = ['fk5','j2000','equatorial']
                gal_coords = ['galactic']

                if verbose: print( "Calculating pixels...")
                if hp_coords in gal_coords:
                    if verbose: print( "Transforming coords...")
                    from astropy.coordinates import SkyCoord
                    import astropy.units as u
                    gc = SkyCoord(ra=ras_deg*u.degree, dec=decs_deg*u.degree, frame='fk5')
                    gc = gc.transform_to('galactic')
                    phOut = gc.l.deg * np.pi/180.
                    thOut = gc.b.deg * np.pi/180.
                    thOut = np.pi/2. - thOut #polar angle is 0 at north pole

                    self.pixs = hp.ang2pix( nside, thOut, phOut )
                elif hp_coords in eq_coords:
                    ras_out = ras_deg
                    decs_out = decs_deg
                    self.pixs = hp.ang2pix(nside,ras_out,decs_out,lonlat=True)

                else:
                    raise ValueError

                if verbose: print( "Done with pixels...")
            else:
                coords = np.vstack((decs_deg,ras_deg))*np.pi/180.
                if verbose: print( "Calculating pixels...")
                self.pixs = enmap.sky2pix(shape,wcs,coords,corner=True) # should corner=True?!
                if verbose: print( "Done with pixels...")
        else:
            self.pixs = pixs

        self.counts = self.get_map(weights=weights)
        if weights is None: 
            self.rcounts = self.get_map(weights=None)
        else:
            self.rcounts = self.counts
        if not self.curved:
            self.counts = enmap.enmap(self.counts,self.wcs)

        self.mask = np.ones(shape) if mask is None else mask
        self._counts()

    def get_map(self,weights=None):
        if self.verbose: print("Calculating histogram...")
        if self.curved:
            return np.histogram(self.pixs,bins=self.shape,weights=weights,range=[0,self.shape],density=False)[0].astype(np.float32)
        else:
            Ny,Nx = self.shape[-2:]
            return enmap.ndmap(np.histogram2d(self.pixs[0,:],self.pixs[1,:],
                                              bins=self.shape,weights=weights,
                                              range=[[0,Ny],[0,Nx]],density=False)[0],self.wcs)

    def _counts(self):
        cts = self.counts.copy()
        cts[self.mask<0.9] = np.nan
        rcts = self.rcounts.copy()
        rcts[self.mask<0.9] = np.nan
        self.ngals = np.nansum(rcts)
        self.nmean = np.nanmean(cts)
        if self.curved:
            area_sqdeg = 4.*np.pi*(180./np.pi)**2.
        else:
            area_sqdeg = enmap.area(self.shape,self.wcs)*(180./np.pi)**2.
        self.frac = self.mask.sum()*1./self.mask.size
        self.area_sqdeg = self.frac*area_sqdeg
        self.ngal_per_arcminsq  = self.ngals/(self.area_sqdeg*60.*60.)

    def get_delta(self):
        delta = (self.counts/self.nmean-1.)
        if not self.curved:
            parea = maps.psizemap(delta.shape, self.wcs)*((180.*60./np.pi)**2.)
            delta = ((delta+1.)*self.nmean/self.ngal_per_arcminsq/parea)-1.
            delta = enmap.enmap(delta,self.wcs)
        return delta
    

def load_boss(boss_files,zmin,zmax,do_weights):
    ras = []
    decs = []
    zs = []
    if do_weights: w = []
    for boss_file in boss_files:
        f = fits.open(boss_file)
        cat = f[1]
        if do_weights: 
            w += (cat.data['WEIGHT_SYSTOT']*(cat.data['WEIGHT_NOZ'] + cat.data['WEIGHT_CP'] - 1.0)).tolist()
        ras += cat.data['RA'].tolist()
        decs += cat.data['DEC'].tolist()
        zs += cat.data['Z'].tolist()
        f.close()
    ras = np.asarray(ras)
    decs = np.asarray(decs)
    zs = np.asarray(zs)
    sel = np.logical_and(zs>=zmin,zs<zmax)
    ras = ras[sel]
    decs = decs[sel]
    if do_weights:
        w = np.asarray(w)
        w = w[sel]
    else:
        w = None
    zs = zs[sel]
    return ras,decs,w


def get_delta(mask,ws=None,ras=None,decs=None,pixs=None,hp_coords='equatorial'):
    assert mask.ndim==1
    npix = mask.size
    nside = hp.npix2nside(npix)

    if pixs is None: 

        eq_coords = ['fk5','j2000','equatorial','equ']
        gal_coords = ['galactic','gal']

        if hp_coords in gal_coords:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            gc = SkyCoord(ra=ras*u.degree, dec=decs*u.degree, frame='fk5')
            gc = gc.transform_to('galactic')
            phOut = gc.l.deg * np.pi/180.
            thOut = gc.b.deg * np.pi/180.
            thOut = np.pi/2. - thOut #polar angle is 0 at north pole
            pixs = hp.ang2pix( nside, thOut, phOut )
        else:
            pixs = hp.ang2pix(nside,ras,decs,lonlat=True)    


    # This gives me n_p = sum_i_in_p w_i
    if ws is None: ws = np.ones(pixs.size)
    wcounts = np.histogram(pixs,bins=npix,weights=ws,range=[0,npix],density=False)[0]
    # the mask should be a number from 0 to 1 that tells me
    # what the area weight for each pixel is
    pix_area = 4*np.pi / npix
    areas = mask * pix_area
    nmean = np.sum(wcounts) / np.sum(areas) # should this be weighted counts
    wall = nmean * areas
    delta = wcounts/wall - 1
    delta[~np.isfinite(delta)] = 0
    fsky = np.sum(areas) / 4. / np.pi
    return delta,fsky
    
    

class BOSSMapper(CatMapper):

    def __init__(self,boss_files,random_files=None,rand_sigma_arcmin=2.,rand_threshold=1e-3,zmin=None,zmax=None,shape=None,wcs=None,nside=None,verbose=True,hp_coords="equatorial",mask=None,do_weights=True):
        from astropy.io import fits

        ras,decs,w = load_boss(boss_files,zmin,zmax,do_weights)
            
        CatMapper.__init__(self,ras,decs,shape,wcs,nside,verbose=verbose,hp_coords=hp_coords,mask=mask,weights=w if do_weights else None)
        if (random_files is not None):
            assert mask is None
            self.rand_map = 0.
            #ras = []
            #decs = []
            for random_file in random_files:
                if verbose: print ("Opening fits...")
                f = fits.open(random_file)
                if verbose: print ("Done opening fits...")
                cat = f[1] #.copy()
                ras = cat.data['RA'] 
                decs = cat.data['DEC'] 
                zs = cat.data['Z'] 
                sel = np.logical_and(zs>=zmin,zs<zmax)
                rcat = CatMapper(ras[sel],decs[sel],shape,wcs,nside,verbose=verbose,hp_coords=hp_coords)
                self.rand_map += rcat.counts
                del rcat
                del ras
                del decs
                del cat
                f.close()
            self.update_mask(rand_sigma_arcmin,rand_threshold)

    def update_mask(self,rand_sigma_arcmin=2.,rand_threshold=1e-3):
        if rand_sigma_arcmin>1.e-3:
            if self.verbose: print( "Smoothing...")
            if self.curved:
                smap = hp.smoothing(self.rand_map,sigma=rand_sigma_arcmin*np.pi/180./60.)
            else:
                smap = enmap.smooth_gauss(self.rand_map,rand_sigma_arcmin*np.pi/180./60.)
            if self.verbose: print( "Done smoothing...")
        else:
            if self.verbose: smap = self.rand_map

        self.mask = np.zeros(self.shape)
        self.mask[smap>rand_threshold] = 1
        if not self.curved:
            self.mask = enmap.enmap(self.mask,self.wcs)
        self._counts()
            
    
class HSCMapper(CatMapper):

    def __init__(self,cat_file=None,pz_file=None,zmin=None,zmax=None,mask_threshold=4.,shape=None,wcs=None,nside=None,hp_coords="equatorial",pzname="mlz",pztype="best"):
        if cat_file[-5:]==".fits":
            f = fits.open(cat_file)
            self.cat = f[1].copy()
            f.close()
        elif cat_file[-4:]==".hdf" or cat_file[-3:]==".h5":
            import h5py,pandas as pd
            df = pd.read_hdf(cat_file)
            class Temp:
                pass
            self.cat = Temp()
            self.cat.data = df
        ras = self.cat.data['ira']
        decs = self.cat.data['idec']
        self.wts = self.cat.data['ishape_hsm_regauss_derived_weight']
        if pz_file is not None:
            fz = fits.open(pz_file)
            self.zs = fz[1].data[pzname+"_photoz_"+pztype]

        CatMapper.__init__(self,ras,decs,shape,wcs,nside,hp_coords=hp_coords)
        self.hsc_wts = self.get_map(weights=self.wts)
        self.mean_wt = np.nan_to_num(self.hsc_wts/self.counts)
        self.update_mask(mask_threshold)

    def update_mask(self,mask_threshold):
        mask = np.zeros(self.shape)
        mask[self.mean_wt>mask_threshold] = 1
        self.mask = mask
        if not self.curved:
            self.mask = enmap.enmap(self.mask,self.wcs)
        self._counts()

        
    def get_shear(self,do_m=True,do_c=True):
        rms = self.cat.data['ishape_hsm_regauss_derived_rms_e']
        m = self.cat.data['ishape_hsm_regauss_derived_bias_m']
        e1 = self.cat.data['ishape_hsm_regauss_e1']
        e2 = self.cat.data['ishape_hsm_regauss_e2']
        c1 = self.cat.data['ishape_hsm_regauss_derived_bias_c1']
        c2 = self.cat.data['ishape_hsm_regauss_derived_bias_c2']

        hsc_wts = self.hsc_wts
        wts = self.wts
        hsc_resp = 1.-np.nan_to_num(self.get_map(weights=(wts*(rms**2.))) / hsc_wts)
        hsc_m = np.nan_to_num(self.get_map(weights=(wts*(m))) / hsc_wts) if do_m else hsc_wts*0.

        hsc_e1 = self.get_map(weights=e1*wts)
        hsc_e2 = self.get_map(weights=e2*wts)

        hsc_c1 = np.nan_to_num(self.get_map(weights=c1*wts)/hsc_wts) if do_c else hsc_wts*0.
        hsc_c2 = np.nan_to_num(self.get_map(weights=c2*wts)/hsc_wts) if do_c else hsc_wts*0.

        g1map = np.nan_to_num(hsc_e1/2./hsc_resp/(1.+hsc_m)/hsc_wts) - np.nan_to_num(hsc_c1/(1.+hsc_m))
        g2map = np.nan_to_num(hsc_e2/2./hsc_resp/(1.+hsc_m)/hsc_wts) - np.nan_to_num(hsc_c2/(1.+hsc_m))

        if not self.curved:
            g1map = enmap.enmap(g1map,self.wcs)
            g2map = enmap.enmap(g2map,self.wcs)
            
        return g1map,g2map

def split_samples(in_samples,split_points):
    """ Calculate statistics on splits of a sample of data.
    If in_samples is a list containing measurements of (say) richness
    and split_points are the richness values corresponding to bin 
    edges of the splits, this function will return the "S/N", 
    average richness and number of objects in each bin defined 
    by the split_points bin edges.

    Here, S/N is average richness times square root of number
    objects in that bin. If the real S/N is linear in richness
    and sqrt(N), this returned quantity should be proportional
    to it.


    e.g. 
    >>> in_samples = [5,1,6,7,2,9,10]
    >>> split_points = [1,5,10]
    >>> sns, means, Ns = split_points(in_samples,split_points)
    >>> print (Ns)
    >>> [3,4]
    

    """
    assert np.all(np.diff(split_points))>0., "Split points should be monotonically increasing."
    A = np.asarray(in_samples)
    sns = []
    means = []
    Ns = []
        
    for a,b in zip(split_points[:-1],split_points[1:]):
        loc = np.where(np.logical_and(A>a,A<=b))
        split_mean = A[loc].mean()
        means.append(split_mean)
        split_N = len(A[loc])
        Ns.append(split_N)
        sns.append(split_mean*np.sqrt(split_N))
        
        
    return np.asarray(sns),np.asarray(means),np.asarray(Ns)


def optimize_splits(in_samples,in_splits):
    """ Given measurements in_samples and bin edges in_splits,
    this function solves for a new set of bin edges out_splits
    (keeping the leftmost and rightmost edge fixed) such
    that the "S/N" in each bin is as close to each other as
    possible. (Concretely, it minimizes the variance of the
    S/N in each bin).

    See split_samples for a definition of S/N.

    """

    def cost(*kwargs):
        in_array = np.asarray(kwargs).flatten()
        if np.any(np.diff(in_array))<0.: return np.inf
        out_splits = np.append(np.append(in_splits[0],in_array),in_splits[-1]).flatten()
        sns,means,Ns = split_samples(in_samples,out_splits)
        return np.var(sns)


    from scipy.optimize import fmin
    in_splits = np.asarray(in_splits)
    res = fmin(cost,in_splits[1:-1])
    out_splits = np.append(np.append(in_splits[0],res),in_splits[-1]).flatten()
    return out_splits


def select_based_on_mask(ras,decs,mask,threshold=0.99):
    """
    Filters ra,dec based on whether it falls within a mask
    """
    coords = np.vstack((decs,ras))*np.pi/180.
    pixs = enmap.sky2pix(mask.shape,mask.wcs,coords).astype(np.int)
    # First select those that fall within geometry
    sel = np.logical_and.reduce([pixs[0]>=0,pixs[1]>=0,pixs[0]<mask.shape[0],pixs[1]<mask.shape[1]])
    pixs = pixs[:,sel]
    coords = coords[:,sel]
    scoords = np.rad2deg(coords[:,mask[pixs[0],pixs[1]]>threshold])
    return scoords[1],scoords[0]


def convert_hilton_catalog_to_enplot_annotate_file(annot_fname,hilton_fits,radius=10,width=4,color='red',mask=None,threshold=0.99):
    convert_fits_catalog_to_enplot_annotate_file(annot_fname,hilton_fits,ra_name='RAdeg',
                                                 dec_name='DECdeg',radius=radius,width=width,
                                                 color=color,mask=mask,threshold=threshold)

def convert_fits_catalog_to_enplot_annotate_file(annot_fname,fits_fname,ra_name='RA',
                                                 dec_name='DEC',radius=10,width=4,
                                                 color='red',mask=None,threshold=0.99):
    cols = load_fits(fits_fname,[ra_name,dec_name])
    ras = cols[ra_name]
    decs = cols[dec_name]
    convert_catalog_to_enplot_annotate_file(annot_fname,ras,
                                            decs,radius=radius,width=width,
                                            color=color,mask=mask,threshold=threshold)

def convert_catalog_to_enplot_annotate_file(annot_fname,ras,
                                            decs,radius=10,width=4,
                                            color='red',mask=None,threshold=0.99):
    if mask is not None: ras,decs = select_based_on_mask(ras,decs,mask,threshold=threshold)
    enplot_annotate(annot_fname,ras,decs,radius,width,color)
    
def enplot_annotate(fname,ras,decs,radius,width,color):
    with open(fname,'w') as f:
        for i,(ra,dec) in enumerate(zip(ras,decs)):
            r = radius[i] if isinstance(radius,list) else radius
            w = width[i] if isinstance(width,list) else width
            c = color[i] if isinstance(color,list) else color
            f.write("c %.4f %.4f 0 0 %d %d %s \n" % (dec,ra,r,w,c))


def hp_from_mangle(weight_ply_files,nside=None,veto_ply_files=None,hp_coords='equ',verbose=False,
                   coords=None,return_coords=False):
    """
    Rasterize a set of mangle ply files to a healpix map.

    Parameters
    ----------

    weight_ply_files : list of strings
        List of filenames of mangle ply files from which weights for the output mask will be
        extracted. The weights from different files will be summed into the same output map,
        so if a pixel is masked in one file, it need not be masked in the final map if a
        different file contains a non-zero weight for it.

    nside : int, optional
        Healpix map nside parameter. If coords is provided, this is optional.

    veto_ply_files : list of strings, optional
        List of filenames of mangle ply files containing regions that need to be masked
        or vetoed. A region only needs to exist in at least one of the provided files
        in order to be masked.

    hp_coords : string, optional
        The coordinate system of the output healpix map. Use 'fk5','j2000','equatorial' or 'equ'
        for Equatorial, which is the default. Use 'galactic' or 'gal' for Galactic.

    verbose : bool, optinal
        Whether to print more information. Defaults to False.

    coords : optional, (2,npix) array
        Pre-calculated ra,dec values corresponding to each pixel in the output healpix map.
        If not provided, nside has to be provided and the coords will be calculated 
        based on the coordinate system hp_coords.

    return_coords : bool
        Whether to return the ra,dec values corresponding to each pixel in the output healpix map.
        Defaults to False.

    Returns
    -------

    output : 1d numpy array
        The resulting healpix map.

    coords : optional, (2,npix) array
        The ra,dec values corresponding to each pixel in the output healpix map.
        Only returned if return_coords is True.

    """

    import pymangle

    if coords is None:
        npix = hp.nside2npix(nside)
        pixs = np.arange(npix,dtype=np.int)
        if verbose: print("Converting healpix pixels to coordinates.")
        ra,dec = hp.pix2ang(nside,pixs,lonlat=True)

        eq_coords = ['fk5','j2000','equatorial','equ']
        gal_coords = ['galactic','gal']

        if hp_coords in gal_coords:
            if verbose: print( "Transforming coords...")
            from astropy.coordinates import SkyCoord
            import astropy.units as u

            gc = SkyCoord(ra*u.degree, dec*u.degree, frame = 'galactic')
            equ = gc.transform_to('fk5')
            ra = equ.ra.deg
            dec = equ.dec.deg
    else:
        ra,dec = coords


    output = 0
    for filename in weight_ply_files:
        if verbose: print(f"Reading weight file {filename}...")
        m=pymangle.Mangle(filename)
        output = output + m.weight(ra,dec)

    if veto_ply_files is None: veto_ply_files = []
    for veto in veto_ply_files:
        if verbose: print(f"Reading veto file {veto}...")
        m=pymangle.Mangle(veto)
        bad = m.contains(ra, dec)
        output[bad] = 0

    if return_coords:
        return output, np.asarray((ra,dec))
    else:
        return output



def df_from_fits(filename,columns=None,rename=None):
    from astropy.table import Table
    table = Table.read(filename)
    df = table.to_pandas()
    if columns is not None: df.drop(df.columns.difference(columns), 1, inplace=True)
    if rename is not None: df.rename(columns=dict(zip(columns, rename)) , inplace=True)
    return df
