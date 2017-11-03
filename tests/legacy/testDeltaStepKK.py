import matplotlib
matplotlib.use('Agg')
from limberTheoryCrossCls import XCorrIntegrator
import time
from mmUtils import Plotter
import numpy as np
from sn import LensForecast, getColNum
from scipy.interpolate import interp1d


startTime = time.time()
ellmax = 3900
myInt = XCorrIntegrator(cosmo={'H0':67.5,'omch2':0.122,'ombh2':0.022,'ns':0.965,'As':2.e-9})


theoryFile = "data/mmTest_scalCovCls.dat"
NlFile = "data/test_mv.csv"
    
shapeNoise = 0.3
bgNgal = 12.
fgNgal = 0.026
    
    

LF = LensForecast()
LF.loadKK(theoryFile,getColNum(0,0,2),NlFile,1,nnorm="none",ntranspower=[0.,1.0])

ells = np.arange(2,ellmax,1)


myInt.addDeltaNz('cmbDelta',myInt.zstar)
myInt.addStepNz('cmbStep1',1050.,1090.)
myInt.addStepNz('cmbStep2',myInt.zstar-2.,myInt.zstar-1.)



print("getting cls..")

ellrange = list(range(2,ellmax,1))
myInt.generateCls(ellrange)
truthCl = myInt.getCl("cmb","cmb")
estCl1 = myInt.getCl("cmbDelta","cmbDelta")
estCl2 = myInt.getCl("cmbStep1","cmbStep1")
estCl3 = myInt.getCl("cmbStep2","cmbStep2")
elapsedTime = time.time() - startTime

print(("Estimation took ", elapsedTime , " seconds."))

    

        
pl = Plotter(scaleY='log',scaleX='log')

cells = LF.theory.gCl("kk",ells)
pl.add(ellrange,truthCl,label="true",ls='-')
pl.add(ellrange,estCl1,label="delta",ls='-')
pl.add(ellrange,estCl2,label="step1",ls='-')
pl.add(ellrange,estCl3,label="step2",ls='-')
pl.add(ells,cells,label="CAMBkk",color='red',ls='--')
pl.legendOn(loc='upper right',labsize=10)
pl.done("output/estcls.png")

pl = Plotter()
for clNow,lab in zip([truthCl,estCl1,estCl2,estCl3],["truth","delta","step 40", "step 1"]):
    intmm = interp1d(ellrange,clNow,bounds_error=False,fill_value=0.)(ells)
    pl.add(ells,intmm/LF.theory.gCl("kk",ells),label=lab)
pl.legendOn(loc='upper right',labsize=10)
pl._ax.set_ylim(0.9,1.1)

pl.done("output/int.png")

for clNow,lab in zip([truthCl,estCl1,estCl2,estCl3],["truth","delta","step 40", "step 1"]):
    intmm = interp1d(ellrange,clNow,bounds_error=False,fill_value=0.)(ells)

    rat = intmm/LF.theory.gCl("kk",ells)
    print((lab,(np.abs(rat[np.logical_and(ells>500.,ells<2000.)]-1.)).max()*100., " %"))
    
