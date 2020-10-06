import os,os.path,shutil,sys
import numpy as np
import pickle
from orphics import cosmology 
try:
    from orphics import cosmology  
except:
    print 'Warning: did not import cosmology!'

###############################################################################
def test_cls_camb(dict_file='orphics/tests/lcdm_baseline.pkl',
                  atol=1e-20, rtol=1e-8):
    """
    This reads from file with a given filename,
    and compares the cls from the file
    with cls created with pars from the same file.
    """

   
    f = open( dict_file, "rb" )
    d = pickle.load( f )
    lmax = d['lmax']
    pars = d['pars']
    cl_base = d['lensed_cls']

    #pars['omch2']= 0.12
    
    cl = {}
    cosmo = cosmology.Cosmology(lmax=lmax,paramDict=pars)
    ells = np.arange(2,lmax-1000)
    cl['ell'] = ells
    cl['tt'] = cosmo.theory.lCl('TT',ells)
    cl['ee'] = cosmo.theory.lCl('EE',ells)
    cl['te'] = cosmo.theory.lCl('TE',ells)
    cl['bb'] = cosmo.theory.lCl('BB',ells)
    
    assert compare_cl_dicts(cl, cl_base, atol=atol,rtol=rtol)
    


###############################################################################
def compare_cl_dicts(cl1, cl2, atol=1e-20,rtol=1e-3):
    """
    compares two cl dictionaries in the format as pkl baseline.
    """
    
    for k,v in cl1.iteritems():
        if k in cl2.keys():
            if np.allclose(v,cl2[k],atol=atol,rtol=rtol):
                pass
            else:
                return False
        else:
            return False
    return True
###############################################################################


