import matplotlib
matplotlib.use('Agg')
from limberTheoryCrossCls import XCorrIntegrator
import time
from mmUtils import Plotter
import numpy as np
from sn import LensForecast, getColNum
from scipy.interpolate import interp1d


ellmax = 3900
myInt = XCorrIntegrator(cosmo={'H0':67.5,'omch2':0.122,'ombh2':0.022,'ns':0.965,'As':2.e-9},numz=100)


    
zsource = 1.0    


myInt.addDeltaNz('delta',zsource)
for i,zwidth in enumerate(np.arange(0.01,0.1,0.01)):
    myInt.addStepNz('step'+str(i),zsource-zwidth/2.,zsource+zwidth/2.)



print("getting cls..")
pl = Plotter(scaleY='log',scaleX='log')

ellrange = list(range(2,ellmax,1))
myInt.generateCls(ellrange)
for i,tag in enumerate(sorted(myInt.kernels.keys())):
    if tag=="cmb": continue
    retcl = myInt.getCl("cmb",tag)
    if tag=="delta":
        compcl = retcl.copy()
        lw=2
        ls="--"
    else:
        lw=1
        ls="-"

    
    pl.add(ellrange,retcl,label=tag,ls=ls,lw=lw)
    rat = (retcl/compcl)
    ratio =   (np.abs(rat[np.logical_and(ellrange>100.,ellrange<2000.)]-1.)).max()*100.
    print((tag, ratio, " %"))
pl.legendOn(loc='upper right',labsize=10)
pl.done("output/estcls.png")
