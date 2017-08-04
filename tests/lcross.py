import orphics.tools.cmb as cmb
import orphics.tools.io as io
import numpy as np
from pyfisher.lensInterface import lensNoise
from ConfigParser import SafeConfigParser 
import argparse


# Get the name of the experiment and lensing type from command line
parser = argparse.ArgumentParser(description='Run a Fisher test.')
parser.add_argument('expName', type=str,help='The name of the experiment in input/params.ini')
parser.add_argument('lensName',type=str,help='The name of the CMB lensing section in input/params.ini. ',default="")
args = parser.parse_args()
expName = args.expName
lensName = args.lensName

# Read config
iniFile = "../pyfisher/input/params.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)


cls = np.loadtxt("data/Nov10_highAcc_CDM_lensedCls.dat")

print cls.shape


ells = np.arange(0,cls.shape[0])
cltt = cls[:,1]

fwhm = 18./60.
rms_noise = 0.1

nls = cmb.noise_func(ells,fwhm,rms_noise)*ells*(ells+1.)/2./np.pi

lcross = ells[nls>cltt][0]
print lcross

pl = io.Plotter(scaleY='log',scaleX='log')
pl.add(ells,cltt)
pl.add(ells,nls)
pl._ax.axvline(x=lcross)
pl.done("cls.png")

lmax = 60000

cambRoot = "data/Nov10_highAcc_CDM"
theory = cmb.loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=lmax)


ls,Nls,ellbb,dlbb,efficiency,cc = lensNoise(Config,expName,lensName,beamOverride=None,noiseTOverride=None,lkneeTOverride=None,lkneePOverride=None,alphaTOverride=None,alphaPOverride=None,gradCut=2000,deg=5.,px=0.1,theoryOverride=theory,bigell=lmax)

ells = np.arange(2,lmax,1)
clkk = theory.gCl("kk",ells)

pl = io.Plotter(scaleY='log',scaleX='log')
pl.add(ells,clkk)
pl.add(ls,Nls)
pl._ax.set_ylim(1.e-12,1.e-5)
pl._ax.set_xlim(10,1e5)
pl.done("clkk.png")
