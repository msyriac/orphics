import camb
from math import pi

class Cosmology(object):
    '''
    A wrapper around CAMB that tries to pre-calculate as much as possible
    Intended to be inherited by other classes like LimberCosmology and 
    ClusterCosmology
    '''
    def __init__(self,paramDict,constDict):

        cosmo = paramDict
        self.paramDict = paramDict
        c = constDict
        self.c = c

        self.c['TCMBmuK'] = self.c['TCMB'] * 1.0e6

        self.H0 = cosmo['H0']
        self.h = self.H0/100.
        try:
            self.omch2 = cosmo['omch2']
        except:
            self.omch2 = (cosmo['om']-cosmo['ob'])*self.H0*self.H0/100./100.
            
        try:
            self.ombh2 = cosmo['ombh2']
        except:
            self.ombh2 = cosmo['ob']*self.H0*self.H0/100./100.
        
        self.pars = camb.CAMBparams()
        self.pars.set_cosmology(H0=self.H0, ombh2=self.ombh2, omch2=self.omch2)
        self.pars.InitPower.set_params(ns=cosmo['ns'],As=cosmo['As'])

        self.results= camb.get_background(self.pars)
        self.omnuh2 = self.pars.omegan * ((self.H0 / 100.0) ** 2.)
        

        self.rho_crit0 = 3. / (8. * pi) * (100 * 1.e5)**2. / c['G_CGS'] * c['MPC2CM'] / c['MSUN_CGS']
        self.cmbZ = 1100.

