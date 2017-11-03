import numpy as np
import orphics.tools.io as io
import orphics.analysis.flatMaps as fmaps
from orphics.theory.cosmology import LimberCosmology
from scipy.interpolate import interp1d

pkzfile = "data/May21_cdm_1.0_cut_ibarrier_iconc.dat"
pkzfile2 = "data/May21_fdm_1.0_cut_ibarrier_iconc.dat"


def getPkz(pkzfile):

    with open(pkzfile,'r') as f:
        t = f.readline()
        zs =  np.array([float(x) for x in t.split()[1:]])

        
    Ps = np.loadtxt(pkzfile)
    H0 = 67.3
    ks = Ps[:,0]*(H0/100.) # 1/Mpc
    Ps = Ps[:,1:]*(2.*np.pi**2)/(ks.reshape([len(ks),1]))**3 # Mpc^3


    return ks,zs,Ps

ks,zs,Pkz = getPkz(pkzfile)

from scipy.interpolate import RectBivariateSpline
pkint = RectBivariateSpline(ks,zs,Pkz,kx=3,ky=3)



ks2,zs2,Pkz2 = getPkz(pkzfile2)


kseval = ks
zseval = zs


io.quickPlot2d(Pkz,"pkz.png")
io.quickPlot2d(pkint(kseval,zseval),"pkzint.png")

lc = LimberCosmology(lmax=8000,pickling=True,kmax=400.,pkgrid_override=pkint)

from alhazen.shear import dndz

step = 0.01
zedges = np.arange(0.,3.0+step,step)
zcents = (zedges[1:]+zedges[:-1])/2.
dndzs = dndz(zcents)


pkz_camb = lc.PK.P(zseval, kseval, grid=True).T
io.quickPlot2d(pkz_camb,"pkz_camb.png")

pl = io.Plotter(scaleY='log',scaleX='log')
pl.add(kseval,pkz_camb[:,0])
pl.add(ks,Pkz[:,0],ls="--")
pl.add(ks2,Pkz2[:,0],ls="--")
pl.done("pk0.png")


cdmfile = "data/May21_matter2lens_WF_CDM_cut_ibarrier_iconc_fCls.csv"
fdmfile = "data/May21_matter2lens_WF_FDM_1.0_cut_ibarrier_iconc_fCls.csv"

lcdm,ckkcdm = np.loadtxt(cdmfile,unpack=True)
lfdm,ckkfdm = np.loadtxt(fdmfile,unpack=True)


lc.addNz("s",zedges,dndzs,bias=None,magbias=None,numzIntegral=300)
lc.addStepNz("g",0.4,0.7,bias=2,magbias=None,numzIntegral=300)


ellrange = np.arange(100,8000,1)
lc.generateCls(ellrange,autoOnly=False,zmin=0.)

clkk = lc.getCl("cmb","cmb")
clss = lc.getCl("s","s")
clsk = lc.getCl("cmb","s")
clsg = lc.getCl("s","g")
clgk = lc.getCl("cmb","g")
clgg = lc.getCl("g","g")


pl = io.Plotter(scaleY='log',scaleX='log')
pl.add(ellrange,clkk,color="C0",label="kk")
pl.add(ellrange,clss,color="C1",label="ss")
pl.add(ellrange,clsk,color="C2",label="sk")
pl.add(ellrange,clgg,color="C3",label="gg")
pl.add(ellrange,clsg,color="C4",label="sg")
pl.add(ellrange,clgk,color="C5",label="gk")



       

pl.add(lcdm,ckkcdm,ls="none",marker="o",markersize=1,color="C0",alpha=0.1)
pl.add(lfdm,ckkfdm,ls="none",marker="x",markersize=1,color="C0",alpha=0.1)


pkint2 = RectBivariateSpline(ks2,zs2,Pkz2,kx=3,ky=3)
lc2 = LimberCosmology(lmax=8000,pickling=True,kmax=400.,pkgrid_override=pkint2)
lc2.addNz("s",zedges,dndzs,bias=None,magbias=None,numzIntegral=300)
lc2.addStepNz("g",0.4,0.7,bias=2,magbias=None,numzIntegral=300)
lc2.generateCls(ellrange,autoOnly=False,zmin=0.)

clkk2 = lc2.getCl("cmb","cmb")
clss2 = lc2.getCl("s","s")
clsk2 = lc2.getCl("cmb","s")
clsg2 = lc2.getCl("s","g")
clgk2 = lc2.getCl("cmb","g")
clgg2 = lc2.getCl("g","g")




from orphics.theory.gaussianCov import LensForecast
LF = LensForecast()
LF.loadSS(ellrange,clsg,30.,shapeNoise=0.3)
pl.add(ellrange,LF.Nls['ss'](ellrange),ls="--")
lcdm = ellrange
cond = np.logical_and(lcdm>100,lcdm<5000)
csscdm = clss[cond]
cssfdm = clss2[cond]
nlss = interp1d(ellrange,LF.Nls['ss'](ellrange))(lcdm[cond])
ells = lcdm[cond]
fsky = 0.4
print(("S/N: ",np.sqrt(np.sum((np.sqrt(fsky*(2.*ells+1.)/2.)*(csscdm-cssfdm)/(csscdm+nlss))**2.))))


lcdm,ckkcdm = np.loadtxt(cdmfile,unpack=True)
lfdm,ckkfdm = np.loadtxt(fdmfile,unpack=True)



pl.add(ellrange,clkk2,color="C0",ls="--")
pl.add(ellrange,clss2,color="C1",ls="--")
pl.add(ellrange,clsk2,color="C2",ls="--")
pl.add(ellrange,clgg2,color="C3",ls="--")
pl.add(ellrange,clsg2,color="C4",ls="--")
pl.add(ellrange,clgk2,color="C5",ls="--")

pl.legendOn(labsize=10)

pl.done("cls.png")


cltestcdm = interp1d(lcdm,ckkcdm)(ellrange)
cltestfdm = interp1d(lfdm,ckkfdm)(ellrange)


def perdiff(a,b):
    return (a-b)*100./b

pl = io.Plotter()

for cl1,cl2,lab in zip([clkk2,clss2,clsk2,clgg2,clsg2,clgk2],[clkk,clss,clsk,clgg,clsg,clgk],["kk","ss","sk","gg","sg","gk"]):
    pl.add(ellrange,perdiff(cl1,cl2),label=lab)

pl.add(ellrange,perdiff(cltestfdm,cltestcdm),label="NamTest",ls="--",color="C0")

pl.legendOn(labsize=10)
pl.done("diff.png")



