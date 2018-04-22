from __future__ import print_function
import numpy as np
from enlib import enmap,resample,bench


def mask_map(imap,iys,ixs,hole_arc,hole_frac=0.6):
    shape,wcs = imap.shape,imap.wcs
    Ny,Nx = shape[-2:]
    px = maps.resolution(shape,wcs)*60.*180./np.pi
    hole_n = int(round(hole_arc/px))
    hole_ny = hole_nx = hole_n
    oshape,owcs = enmap.geometry(pos=(0.,0.),shape=(2*hole_n,2*hole_n),res=px*np.pi/180./60.)
    modrmap = enmap.modrmap(oshape,owcs)
    mask = enmap.ones(shape,wcs)
    
    for iy,ix in zip(iys,ixs):
        if iy<=hole_n or ix<=hole_n or iy>=(Ny-hole_n) or ix>=(Nx-hole_n): continue
        vslice = imap[np.int(iy-hole_ny):np.int(iy+hole_ny),np.int(ix-hole_nx):np.int(ix+hole_nx)]
        if np.any(vslice.shape!=oshape): continue
        vslice[modrmap<(hole_frac*hole_arc)*np.pi/180./60.] = np.nan # !!!! could cause a bias
        mask[np.int(iy-hole_ny):np.int(iy+hole_ny),np.int(ix-hole_nx):np.int(ix+hole_nx)][modrmap<hole_arc*np.pi/180./60.] = 0
        
    return mask



def inpaint_map(imap,ras,decs,radii_tags,radii_dict,tot_power_2d,seed=None):
    """
    Brute-force inpaints a map in circular regions.

    imap -- (Ny,Nx) enmap
    ras  -- list of RA of centers in degrees, length M
    decs -- list of DEC of centers in degrees, length M
    radii -- list of strings specifying radius tag for each object, length M
    radii_dict -- dict mapping radius tag string to float value of radius in arcminutes, length K
                  Most expensive operation scales with K.
    tot_power_2d -- total power in map in physical units (e.g. uK^2 radians)

    """


    class G:
        pass

    geometries = {}
    tags = radii_dict.keys()
    for key in tags:
        geometries[key] = calculate_circular_geometry(shape,wcs)




def prepare_circular_mask(stamp,hole_arcminutes):
    '''This routine accepts a stamp as a 2D array. The region
    within hole_arcminutes is masked, and the mean of the region outside it
    is subtracted from the stamp. It returns the masked stamp and the value
    of the mean of the unmasked region that needs to be added back after
    inpainting.

    '''

    stamp[stamp.modrmap()<hole_arcminutes*np.pi/180./60.] = np.nan
    unmasked_mean = np.nanmean(stamp)
    stamp -= unmasked_mean

    return stamp, unmasked_mean


def get_geometry_shapes(shape,wcs,hole_arcminutes):
    modrmap = enmap.modrmap(shape,wcs)
    m1 = np.where(modrmap.reshape(-1)<hole_arcminutes*np.pi/180./60.)[0]
    m2 = np.where(modrmap.reshape(-1)>=hole_arcminutes*np.pi/180./60.)[0]
    return m1.size,m2.size




def get_geometry(pixcov,m1,m2):
    # m1 is hole
    # m2 is context

    Cinv = np.linalg.inv(pixcov)
    # Apply woodbury! to ones everywhere mean or better yet horizontal stripes for each row of the stamp. Correlation length along x can be completely large. Subtracting mean of unmasked region unnecessary.
    cslice = Cinv[m1][:,m1]
    meanMul1 = np.linalg.inv(cslice)

    mul2 = Cinv[m1][:,m2]
    meanMul = np.dot(-meanMul1,mul2)
    cov = np.linalg.pinv(Cinv[m1][:,m1])
    return meanMul, cov


def fill_hole(masked_stamp,meanMatrix,holeArc,m1,m2,covRoot=None):
    '''Returns the result of an inpaint operation as a 1d unraveled vector

    Arguments
    ---------

    masked_liteMap_stamp - the cutout stamp that contains a masked hole and
                           unmasked context
    meanMatrix           - an (nh,nc) matrix. See docs for make_circular_geometry
    holeArc              - radius of masked hole in arcminutes
    m1                   - a 1d boolean selecting the hole region on an unraveled stamp
    m2                   - a 1d boolean selecting the context region on an unraveled stamp
    covRoot              - the square root of the covariance matrix inside the hole. See
                           docs for make_circular_geometry. If unspecified, the random
                           realization returned is zero.

    Returns
    -------

    mean  -  a 1d (nh) vector containing the mean inpainted value constrained according
             to the context
    rand  -  a 1d (nh) vector containing a random realization inside the hole
    sim   -  a 1d (nh) vector containing the sum of mean and rand

    '''

    mean = np.dot(meanMatrix,masked_stamp.reshape(-1)[m2])
    r = np.random.normal(0.,1.,size=(m1.size))
    if covRoot is not None:
        rand = np.dot(covRoot,r)
    else:
        rand = 0.
    sim = mean + rand
    return mean, rand, sim

                                                      



def fill_map(imap,iys,ixs,hole_arc,mean_mul,cov_root,m1,tshape,twcs,seed=None):
    Ny,Nx = imap.shape[-2:]
    sny,snx = tshape[-2:]
    modrmap = enmap.modrmap(tshape,twcs)
    ttemplate = enmap.empty(tshape,twcs)

    iys = iys.astype(np.int)
    ixs = ixs.astype(np.int)

    m1 = np.where(modrmap.reshape(-1)<hole_arc*np.pi/180./60.)[0]
    m2 = np.where(modrmap.reshape(-1)>=hole_arc*np.pi/180./60.)[0]    
    if seed is not None: np.random.seed(seed)

    # Further improvement possible by pre-calculating random vectors
    
    outside = 0
    j = 0
    for i,(iy,ix) in enumerate(zip(iys,ixs)):

        sy = iy-sny/2
        ey = iy+sny/2
        sx = ix-snx/2
        ex = ix+snx/2
        oslice = imap[sy:ey,sx:ex]

        if np.any(oslice.shape!=tshape) or sy<0 or sx<0 or ey>=Ny or ex>=Nx:
            outside+=1
            continue

        j += 1
        ttemplate = oslice.copy()
        
        masked, maskedMean = prepare_circular_mask(ttemplate,hole_arc)
                
        masked = np.nan_to_num(masked)
        mean, rand, sim = fill_hole(masked,mean_mul,hole_arc,m1,m2,cov_root)


        a = masked.reshape(-1)
        # a[m1] = sim+maskedMean
        a[m1] = mean+maskedMean
        a[m2] = oslice.reshape(-1)[m2]
        oslice[:,:] = a.reshape(masked.shape)

        
        
    if outside>0: print (outside, " pt source(s) at edge.")





    
def paste(targetTemplate,m,pasteThis):
    '''Paste the result of an inpaint operation into a rectangular
    np array in a liteMap

    targetTemplate  - a liteMap with the shape of your cutout stamp
    m               - a 1d boolean array that specifies where in the unraveled
                      cutout stamp the hole is
    pasteThis - the result of an inpaint operation, say from fill_hole
    '''
    a = targetTemplate.copy()
    a.reshape(-1)[m] = pasteThis
    a.reshape(targetTemplate.shape)
    return a
