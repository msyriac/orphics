from orphics.theory.limber import XCorrIntegrator

cosmo = {}
cosmo['omch2'] = 0.1194
cosmo['ombh2'] = 0.022
cosmo['H0'] = 67.0
cosmo['ns'] = 0.96
cosmo['As'] = 2.2e-9
#mnu = 0.06,0.02
#w0 = -1.0,0.3

x = XCorrIntegrator(cosmo)
x.addDeltaNz("jiagalaxy",zsource=2.0)

import numpy as np        

print((-np.trapz(x.kernels['jiagalaxy']['W'],x.zs)))
