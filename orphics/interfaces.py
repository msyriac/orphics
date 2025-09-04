from __future__ import print_function
import os, sys, glob
from tempfile import mkstemp
from shutil import move, copyfile
from os import remove, close
import subprocess
import numpy as np
import healpy as hp

# Python 2/3 compatibility
try:
    basestring
except NameError:
    basestring = str


def agora_redshift_to_halocat_files(z_min, z_max, lensed=False):
    from astropy.cosmology import FlatLambdaCDM
    from astropy import units as u

    if lensed:
        base_filename = 'agora_halos_lenra_lendec_mag_rotreal_rotimag_deflectnside16384_{}.npy'
    else:
        base_filename='agora_halolc_rot_{}_v050223.npz'
    
    # Convert redshift to comoving distance in Mpc/h
    cosmo = FlatLambdaCDM(H0=67.77, Om0=0.307, Ob0=0.048) # Agora cosmology
    d_min = cosmo.comoving_distance(z_min).to(u.Mpc).value * cosmo.h
    d_max = cosmo.comoving_distance(z_max).to(u.Mpc).value * cosmo.h

    # Each slice is 25 Mpc/h thick, and slice numbers start from 4,
    # but add buffer in each direction
    slice_start = max(int(d_min // 25) - 1,4)
    slice_end = min(int(d_max // 25) + 1,200)

    # Generate list of corresponding filenames
    filenames = [base_filename.format(i) for i in range(slice_start, slice_end + 1)]
    return filenames



def get_agora_halos(z_min = 0.30, z_max = 0.35,
                    mass_min = 3e14, mass_max = 3.3e14,
                    mmap_mode='r',lensed=True,
                    verbose=False,massdef='m500',
                    lensed_croot='/data5/sims/agora_sims/full/halocat_lensed',
                    unlensed_croot='/data5/sims/agora_sims/full/halocat'):

    
    lfiles = agora_redshift_to_halocat_files(z_min, z_max,lensed=True)
    ufiles = agora_redshift_to_halocat_files(z_min, z_max,lensed=False)
    oras = []
    odecs = []
    ozs = []
    oms = []


    for lcat,ucat in zip(lfiles,ufiles):

        if verbose: print(" ::: loading agora:", cat)
        dat = np.load(f"{unlensed_croot}/{ucat}",mmap_mode=mmap_mode)
        ldat = np.load(f"{lensed_croot}/{lcat}",mmap_mode=mmap_mode)

        
        mh = dat[f'tot{massdef}']

        if lensed:
            ras = ldat[:,0]
            decs = ldat[:,1]
        else:
            ras = dat['totra']
            decs = dat['totdec']
            
        zs = dat['totz']
        h = 0.6777 # Agora cosmology
        Mx00c = mh / h # convert to Msun 

        keep = (Mx00c > mass_min) & (Mx00c < mass_max) & (zs > z_min) & (zs < z_max)
        ras = ras[keep] 
        decs = decs[keep]
        zs = zs[keep]
        masses = Mx00c[keep]

        oras = np.append(oras,ras)
        odecs = np.append(odecs,decs)
        ozs = np.append(ozs,zs)
        oms = np.append(oms,masses)

    return oras,odecs,ozs,oms
    

def load_sdss_redmapper(path, lams=True, zs=True):
    from orphics import catalogs

    extra = []
    if lams:
        extra += ["LAMBDA"]
    if zs:
        extra += ["Z_LAMBDA"]
    return catalogs.load_fits(
        "%s/redmapper_dr8_public_v6.3_catalog.fits" % path,
        column_names=["RA", "DEC"] + extra,
    )




class WebSkySlicer(object):
    def __init__(
        self, dirpath, npatches=72, height_deg=10.0, px_arcmin=2.0, cache_alms=True
    ):
        assert npatches % 2 == 0
        width_deg = 360.0 / (npatches / 2)
        self.geoms = []
        for i in range(npatches):
            box = np.deg2rad([[0, i * width_deg], [height_deg, (i + 1) * width_deg]])
            shape, wcs = enmap.geometry(pos=box, res=np.deg2rad(px_arcmin / 60.0))
            self.geoms.append((shape, wcs))
            box = np.deg2rad([[-height_deg, i * width_deg], [0, (i + 1) * width_deg]])
            shape, wcs = enmap.geometry(pos=box, res=np.deg2rad(px_arcmin / 60.0))
            self.geoms.append((shape, wcs))
        self.dirpath = dirpath
        self._cache = cache_alms
        self._alm_cache = {}
        self.npatches = npatches

    def _load_map(self, path, index, lmax, tag):
        shape, wcs = self.geoms[index]
        try:
            palm = self._alm_cache[tag]
        except:
            palm = None
        imap, alms = reproject.enmap_from_healpix(
            self.dirpath + path,
            shape,
            wcs,
            lmax=lmax,
            rot=None,
            alm=palm,
            return_alm=True,
        )
        if self._cache:
            self._alm_cache[tag] = alms.copy()
        return imap

    def get_y(self, index, lmax=6000):
        ymap = self._load_map("tsz/compton-y.fits", index, lmax, "y")
        return ymap

    def get_tsz(self, index, freq_ghz, lmax=6000):
        return self.get_y(index, lmax=lmax) * 2.7255e6 * fgs.ffunc(freq_ghz)

    def get_cib(self, index, freq_ghz, lmax=6000, halo=True, field=True):
        freqs = np.array([143, 217, 353, 545, 857])  # available frequencies
        freq = freqs[(np.abs(freqs - freq_ghz)).argmin()]  # find closest frequency
        scaling = fgs.cib_nu(freq_ghz) / fgs.cib_nu(freq)  # apply scaling
        cib = 0.0
        if halo:
            cib += self._load_map("cib/%d-halo.fits" % freq, index, lmax, "%d-halo")
        if field:
            cib += self._load_map("cib/%d-field.fits" % freq, index, lmax, "%d-field")
        return cib * scaling * 1e6 / fgs.JyPerSter_to_dimensionless(freq_ghz) * 2.7255e6

    def get_ksz(self, index, lmax=6000, halo=True, field=True):
        ksz = 0.0
        if halo:
            ksz += self._load_map("ksz/ksz-halo.fits", index, lmax, "ksz-halo")
        if field:
            ksz += self._load_map("ksz/ksz-field.fits", index, lmax, "ksz-field")
        return ksz * 2.7255e6

    def get_kappa(self, index, lmax=6000):
        return self._load_map("tcmb/kappa.fits", index, lmax, "kappa")

    def get_cmb(self, index, lensed=True, lmax=6000):
        lstring = "" if lensed else "un"
        return (
            self._load_map(
                "tcmb/tcmb_%slensed_alms.fits" % lstring,
                index,
                lmax,
                "cmb_%slensed" % lstring,
            )
            * 1e6
        )


def websky_halos(dirpath="./", mmin=-np.inf, mmax=np.inf):

    pkfile = open(dirpath + "halos/halos.pksc", "rb")
    Nhalo = np.fromfile(pkfile, dtype=np.int32, count=1)[0]
    RTHMAXin = np.fromfile(pkfile, dtype=np.float32, count=1)
    redshiftbox = np.fromfile(pkfile, dtype=np.float32, count=1)

    nfloats_perhalo = 10
    npkdata = nfloats_perhalo * Nhalo
    peakdata = np.fromfile(pkfile, dtype=np.float32, count=npkdata)
    peakdata = np.reshape(peakdata, (Nhalo, nfloats_perhalo))

    Rth = peakdata[:, 6]
    Omega_M = 0.25
    h = 0.7
    rho = 2.775e11 * Omega_M * h ** 2
    M = 4.0 / 3 * np.pi * Rth ** 3 * rho
    sel = np.logical_and(M > mmin, M <= mmax)

    xpos = peakdata[:, 0][sel]
    ypos = peakdata[:, 1][sel]
    zpos = peakdata[:, 2][sel]
    vzpos = peakdata[:, 5][sel]
    h = h
    M = M
    from orphics import cosmology

    vecs = np.swapaxes(np.array([xpos, ypos, zpos]), 0, 1)
    ras, decs = hp.vec2ang(vecs, lonlat=True)
    params = cosmology.defaultCosmology
    params["H0"] = 70.0
    params["omch2"] = 0.10331
    params["ombh2"] = 0.01919
    cc = cosmology.Cosmology(params, skipCls=True, skipPower=True, skip_growth=True)
    cspeed = 2.9979458e8 / 1e3
    chis = np.sqrt(xpos ** 2.0 + ypos ** 2.0 + zpos ** 2.0)
    zs = cc.results.redshift_at_comoving_radial_distance(chis) + vzpos / cspeed
    return ras, decs, zs


def sehgal_halos(
    halo_file="./../sehgal/halo_nbody_m200mean.hd5",
    mmin=-np.inf,
    mmax=np.inf,
    zmin=-np.inf,
    zmax=np.inf,
):
    import pandas as pd

    df = pd.read_hdf(halo_file)

    ra = df["RA"] * np.pi / 180.0
    dec = df["DEC"] * np.pi / 180.0

    new_dfs = []

    for n in range(0, 4):
        new_df = df.copy()
        new_df["RA"] = (ra + n * np.pi / 2.0) * 180.0 / np.pi
        new_dfs.append(new_df.copy())

    for n in range(4):
        new_df = df.copy()
        new_df["RA"] = (np.pi / 2.0 - ra + n * np.pi / 2.0) * 180.0 / np.pi
        new_df["DEC"] = -dec * 180.0 / np.pi
        new_dfs.append(new_df.copy())

    final_df = pd.concat(new_dfs)
    assert final_df["RA"].size == 8 * df["RA"].size

    cond = np.abs(final_df["RA"]) >= 0.0
    cond2 = np.abs(final_df["DEC"]) >= 0.0
    sel = cond2 & cond
    final_df = final_df[sel]
    df = final_df

    sel = (
        (df["Z"] > zmin) & (df["Z"] < zmax) & (df["M200"] > mmin) & (df["M200"] < mmax)
    )

    df = df[sel]

    zs = df["Z"].values
    m200s = df["M200"].values
    ras = df["RA"].values
    decs = df["DEC"].values

    return ras, decs, zs


class PlanckLensing(object):
    def __init__(
        self, froot="/gpfs01/astro/workarea/msyriac/data/planck/pr3/", nside=2048
    ):

        self.nside = nside
        self.froot = froot

    def _get_real(self, ifile):
        from orphics.maps import filter_alms

        alm = hp.read_alm(ifile)
        alm = filter_alms(alm, 8, 2048)
        omap = hp.alm2map(alm, nside=self.nside)
        return omap

    def load_planck_lensing(self, tsz_deproj=False, pr2=False, est="MV", inhom=False):
        assert est in ["TT", "PP", "MV"]
        if tsz_deproj:
            pdir = self.froot + "COM_Lensing_Szdeproj_4096_R3.00/TT/"
        elif inhom:
            pdir = self.froot + "COM_Lensing_Inhf_2048_R3.00/%s/" % est
        else:
            pdir = self.froot + "COM_Lensing_4096_R3.00/%s/" % est

        if pr2:
            pdir = self.froot + "../archived/lensing/"

        omap = self._get_real(pdir + "dat_klm.fits")

        return omap

    def load_mask(self):
        imask = hp.read_map(self.froot + "COM_Lensing_4096_R3.00/mask.fits.gz")
        inside = hp.npix2nside(imask.size)
        print("mask nside ; ", inside)
        if inside != self.nside:
            imask = hp.ud_grade(imask, nside_out=self.nside)
        return imask


# TODO:
# test that different MPI jobs get different temp ini files


class CAMBInterface(object):
    """ Credit: A lot of this was written by Nam Ho Nguyen
    This interface lets you call Fortran CAMB from Python with full control over the ini parameters
    and retrieve galaxy Cls. It is primarily written with CAMB Sources in mind. TODO: Generalize

    """

    def __init__(self, ini_template, camb_loc):
        """
        ini_template is the full path to a file that will be used as the "base" ini file. Parameters can be
        modified and added relative to this ini. Usually this is the barebones "params.ini" or "params_lensing.ini"
        in the root of CAMB.

        camb_loc is the path to the directory containing the "camb" executable.

        """

        # cp ini_template to temporary
        self.ifile = ini_template.strip()[:-4] + "_itemp_" + str(os.geteuid()) + ".ini"
        copyfile(ini_template, self.ifile)
        self.out_name = "itemp_" + str(os.geteuid())
        self.set_param("output_root", self.out_name)

        self.camb_loc = camb_loc

    def set_param(self, param, value):
        """ Set a parameter to a certain value. If it doesn't exist, it appends to the end of the base ini file.
        """
        self._replace(self.ifile, param, subst=param + "=" + str(value))

    def call(self, suppress=True):
        """
        Once you're done setting params, just use the call() function to run CAMB.
        Set suppress = False to get the full CAMB output.
        """
        if suppress:
            with open(os.devnull, "w") as f:
                subprocess.call(
                    [self.camb_loc + "/camb", self.ifile], stdout=f, cwd=self.camb_loc
                )
        else:
            subprocess.call([self.camb_loc + "/camb", self.ifile], cwd=self.camb_loc)

    def get_cls(self):
        """
        This function returns the Cls output by CAMB Sources. TODO: Generalize.
        If there are N redshift windows specified, then this returns:
        ells, clarr
        where clarr is of shape (N+3,N+3,ells.size)

        ells.size is determined by the lmax requested in the ini file
        The [i,j,:] slice is the cross-correlation of the ith component with the jth
        component in L(L+1)C/2pi form. (It is symmetric and hence redundant in i,j)
        
        The order of the components is:
        CMB T
        CMB E
        CMB phi
        redshift1
        redshift2
        ... etc.

        What the redshift components corresponds to (galaxy overdensities or galaxy lensing) 
        depends on the ini file and the set parameters. 
        """

        filename = self.camb_loc + "/" + self.out_name + "_scalCovCls.dat"
        clarr = np.loadtxt(filename)
        ells = clarr[:, 0]
        ncomps = int(np.sqrt(clarr.shape[1] - 1))
        assert ncomps ** 2 == (clarr.shape[1] - 1)
        cls = np.swapaxes(clarr[:, 1:], 0, 1)
        return ells, cls.reshape((ncomps, ncomps, ells.size))

    def _replace(self, file_path, pattern, subst):
        # Internal function
        flag = False
        fh, abs_path = mkstemp()
        with open(abs_path, "w") as new_file:
            with open(file_path) as old_file:
                for line in old_file:
                    # if pattern in line:
                    if "".join(line.split())[: len(pattern) + 1] == (pattern + "="):
                        line = subst + "\n"
                        flag = True
                    new_file.write(line)
                if not (flag) and ("transfer_redshift" in pattern):
                    line = subst + "\n"
                    flag = True
                    new_file.write(line)

            if not (flag):
                line = "\n" + subst + "\n"
                new_file.write(line)

        close(fh)
        remove(file_path)
        move(abs_path, file_path)

    def __del__(self):
        remove(self.ifile)


def test():
    # Demo
    citest = CAMBInterface("params_test.ini", ".")
    citest.set_param("num_redshiftwindows", "3")
    citest.set_param("redshift(3)", "2")
    citest.set_param("redshift_kind(3)", "lensing")
    citest.set_param("redshift_sigma(3)", "0.03")
    citest.call(suppress=False)
    ells, cls = citest.get_cls()
    print(cls.shape)
