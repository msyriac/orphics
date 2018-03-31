'''
Utilities for dealing with galaxy catalogs, projecting catalogs into pixelated maps, etc.
'''


import numpy as np
from enlib import enmap, coordinates
import healpy as hp
from astropy.io import fits

def load_fits(fits_file,column_names,hdu_num=1):
    hdu = fits.open(fits_file)
    columns = {}
    for col in column_names:
        columns[col] = hdu[hdu_num].data[col]
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
    """
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

    def __init__(self,ras_deg,decs_deg,shape=None,wcs=None,nside=None,verbose=True,hp_coords="equatorial"):

        self.verbose = verbose
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
                lonlat = True
                self.pixs = hp.ang2pix(nside,ras_out,decs_out,lonlat=lonlat)
                
            else:
                raise ValueError
                
            if verbose: print( "Done with pixels...")
            self.nside = nside
            self.shape = hp.nside2npix(nside)
            self.curved = True
        else:
            coords = np.vstack((decs_deg,ras_deg))*np.pi/180.
            self.shape = shape
            self.wcs = wcs
            if verbose: print( "Calculating pixels...")
            self.pixs = enmap.sky2pix(shape,wcs,coords,corner=True) # should corner=True?!
            if verbose: print( "Done with pixels...")
            self.curved = False
        self.counts = self.get_map()
        if not self.curved:
            self.counts = enmap.enmap(self.counts,self.wcs)

        self.mask = np.ones(shape)
        self._counts()

    def get_map(self,weights=None):
        if self.verbose: print("Calculating histogram...")
        if self.curved:
            return np.histogram(self.pixs,bins=self.shape,weights=weights,range=[0,self.shape])[0].astype(np.float32)
        else:
            Ny,Nx = self.shape
            return enmap.ndmap(np.histogram2d(self.pixs[0,:],self.pixs[1,:],bins=self.shape,weights=weights,range=[[0,Ny],[0,Nx]])[0],self.wcs)

    def _counts(self):
        cts = self.counts.copy()
        cts[self.mask<0.9] = np.nan
        self.ngals = np.nansum(cts)
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
            delta = enmap.enmap(delta,self.wcs)
        return delta
    


class BOSSMapper(CatMapper):

    def __init__(self,boss_files,random_files=None,rand_sigma_arcmin=2.,rand_threshold=1e-3,zmin=None,zmax=None,shape=None,wcs=None,nside=None,verbose=True,hp_coords="equatorial"):
        from astropy.io import fits

        ras = []
        decs = []
        for boss_file in boss_files:
            f = fits.open(boss_file)
            cat = f[1] #.copy()
            ras += cat.data['RA'].tolist()
            decs += cat.data['DEC'].tolist()
            f.close()
            
        CatMapper.__init__(self,ras,decs,shape,wcs,nside,verbose=verbose,hp_coords=hp_coords)
        if random_files is not None:
            self.rand_map = 0.
            #ras = []
            #decs = []
            for random_file in random_files:
                if verbose: print ("Opening fits...")
                f = fits.open(random_file)
                if verbose: print ("Done opening fits...")
                cat = f[1] #.copy()
                ras = cat.data['RA'] #.tolist()
                decs = cat.data['DEC'] #.tolist()
                rcat = CatMapper(ras,decs,shape,wcs,nside,verbose=verbose,hp_coords=hp_coords)
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

    def __init__(self,cat_file=None,pz_file=None,zmin=None,zmax=None,mask_threshold=4.,shape=None,wcs=None,nside=None):
        if cat_file[-5:]==".fits":
            from astropy.io import fits
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
            raise NotImplementedError

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
