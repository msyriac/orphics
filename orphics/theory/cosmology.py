import camb
from math import pi
from orphics.tools.cmb import loadTheorySpectraFromPycambResults

class Cosmology(object):
    '''
    A wrapper around CAMB that tries to pre-calculate as much as possible
    Intended to be inherited by other classes like LimberCosmology and 
    ClusterCosmology
    '''
    def __init__(self,paramDict,constDict,lmax):

        cosmo = paramDict
        self.paramDict = paramDict
        c = constDict
        self.c = c

        self.c['TCMBmuK'] = self.c['TCMB'] * 1.0e6

        self.H0 = cosmo['H0']
        self.h = self.H0/100.
        try:
            self.omch2 = cosmo['omch2']
            self.om = (cosmo['omch2']+cosmo['ombh2'])/self.h**2.
        except:
            self.omch2 = (cosmo['om']-cosmo['ob'])*self.H0*self.H0/100./100.
            
        try:
            self.ombh2 = cosmo['ombh2']
            self.ob = cosmo['ombh2']/self.h**2.
        except:
            self.ombh2 = cosmo['ob']*self.H0*self.H0/100./100.
        
        self.mnu = cosmo['mnu']
        self.w0 = cosmo['w0']

        self.pars = camb.CAMBparams()
        self.pars.set_cosmology(H0=self.H0, ombh2=self.ombh2, omch2=self.omch2, mnu=self.mnu,) # add tau
        self.pars.set_dark_energy(w=self.w0)
        self.pars.InitPower.set_params(ns=cosmo['ns'],As=cosmo['As'])

        self.results= camb.get_background(self.pars)
        self.omnuh2 = self.pars.omegan * ((self.H0 / 100.0) ** 2.)
        

        self.rho_crit0H100 = 3. / (8. * pi) * (100 * 1.e5)**2. / c['G_CGS'] * c['MPC2CM'] / c['MSUN_CGS']
        self.cmbZ = 1100.

        print "Generating theory Cls..."
        self.pars.set_accuracy(AccuracyBoost=2.0, lSampleBoost=4.0, lAccuracyBoost=4.0)
        self.pars.set_for_lmax(lmax=(lmax+500), lens_potential_accuracy=3, max_eta_k=2*(lmax+500))
        theory = loadTheorySpectraFromPycambResults(self.results,self.pars,lmax,unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=9000)
        self.clttfunc = lambda ell: theory.lCl('TT',ell)
        self.theory = theory
    
