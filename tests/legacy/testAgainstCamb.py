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
LF.loadSS(theoryFile,getColNum(1,1,2),shapeNoise=shapeNoise,ngal=bgNgal)
LF.loadKS(theoryFile,getColNum(0,1,2))
LF.loadGG(theoryFile,getColNum(2,2,2),ngal=fgNgal)
LF.loadKG(theoryFile,getColNum(0,2,2))
LF.loadSG(theoryFile,colnums=[getColNum(1,2,2),getColNum(2,1,2)])

ells = np.arange(2,ellmax,1)


from scipy.stats import norm
print("making dndz")
zmin = 0.
zmax = 5.0
zstep = 0.1
zmids = np.arange(zmin+zstep/2.,zmax+zstep/2.,zstep)
zpass = np.arange(zmin,zmax+zstep,zstep)
dndzpass = norm.pdf(zmids,loc=1.0,scale=0.2)
myInt.addNz('s',zpass,dndzpass)#,diagnostic=[0.8,1.2,10])

czedges, cdndz = np.loadtxt("../cmb-lensing-projections/data/dndz/cmass_dndz.csv",delimiter=' ',unpack=True)
cdndz = cdndz[1:]
myInt.addNz('g',czedges,cdndz,bias=2.0,magbias=0.42)#,diagnostic=[0.8,1.2,10])


print("getting cls..")

ellrange = list(range(2,ellmax,1))
myInt.generateCls(ellrange)
retclkks = myInt.getCl("cmb","cmb")
retclkss = myInt.getCl("cmb","s")
retclsss = myInt.getCl("s","s")
retclsks = myInt.getCl("s","cmb")
retclkgs = myInt.getCl("cmb","g")
retclggs = myInt.getCl("g","g")
retclgks = myInt.getCl("g","cmb")
elapsedTime = time.time() - startTime

print(("Estimation took ", elapsedTime , " seconds."))

    

        
pl = Plotter(scaleY='log',scaleX='log')

cells = LF.theory.gCl("kk",ells)
pl.add(ellrange,retclkks,label="MMkk",color='red',ls='-')
pl.add(ells,cells,label="CAMBkk",color='red',ls='--')

cells = LF.theory.gCl("ks",ells)
pl.add(ellrange,retclkss,label="MMks",color='blue',ls='-')
pl.add(ellrange,retclsks,label="MMsk",color='blue',ls='-.',lw=2)
pl.add(ells,cells,label="CAMBks",color='blue',ls='--')

cells = LF.theory.gCl("ss",ells)
pl.add(ellrange,retclsss,label="MMss",color='green',ls='-')
pl.add(ells,cells,label="CAMBss",color='green',ls='--')


cells = LF.theory.gCl("kg",ells)
pl.add(ellrange,retclkgs,label="MMkg",color='purple',ls='-')
pl.add(ellrange,retclgks,label="MMgk",color='purple',ls='-.',lw=2)
pl.add(ells,cells,label="CAMBkg",color='purple',ls='--')

cells = LF.theory.gCl("gg",ells)
pl.add(ellrange,retclggs,label="MMgg",color='orange',ls='-')
pl.add(ells,cells,label="CAMBgg",color='orange',ls='--')

pl.legendOn(loc='upper right',labsize=10)
pl.done("output/estcls.png")


intmmks = interp1d(ellrange,retclkss,bounds_error=False,fill_value=0.)(ells)
intmmkk = interp1d(ellrange,retclkks,bounds_error=False,fill_value=0.)(ells)
intmmss = interp1d(ellrange,retclsss,bounds_error=False,fill_value=0.)(ells)
intmmkg = interp1d(ellrange,retclkgs,bounds_error=False,fill_value=0.)(ells)
intmmgg = interp1d(ellrange,retclggs,bounds_error=False,fill_value=0.)(ells)
pl = Plotter()
pl.add(ells,intmmks/LF.theory.gCl("ks",ells))
pl.add(ells,intmmkk/LF.theory.gCl("kk",ells))
pl.add(ells,intmmss/LF.theory.gCl("ss",ells))
pl.add(ells,intmmkg/LF.theory.gCl("kg",ells))
pl.add(ells,intmmgg/LF.theory.gCl("gg",ells))
pl.legendOn(loc='upper right',labsize=10)
pl._ax.set_ylim(0.,1.5)
pl.done("output/int.png")

ratkk = intmmkk/LF.theory.gCl("kk",ells)
ratks = intmmks/LF.theory.gCl("ks",ells)
ratkg = intmmkg/LF.theory.gCl("kg",ells)

for ellcut in [2,100,200,300,500]:
    print(("kk",ellcut,(np.abs(ratkk[ells>ellcut]-1.)).max()*100., " %"))
    print(("ks",ellcut,(np.abs(ratks[ells>ellcut]-1.)).max()*100., " %"))
    print(("kg",ellcut,(np.abs(ratkg[ells>ellcut]-1.)).max()*100., " %"))



weff = myInt.kernels['cmb']['W']
wgal = myInt.kernels['s']['W']
wgalc = myInt.kernels['g']['W']

pl = Plotter(scaleX='log')
pl.add(myInt.zs,weff/weff.max(),label="cmb lens kernel")
pl.add(zmids,dndzpass/dndzpass.max(),label="gal dndz")
pl.add(czedges[1:],cdndz/cdndz.max(),label="spec dndz")
pl.add(myInt.zs,wgal/wgal.max(),label="gal lens kernel")
pl.add(myInt.zs,wgalc/wgalc.max(),label="spec kernel")
pl.legendOn(loc='upper right',labsize=10)
pl.done("output/weff.png")
