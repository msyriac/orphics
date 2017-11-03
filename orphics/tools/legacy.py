from scipy.interpolate import splrep,splev
import numpy as np

import re
from orphics.tools.io import bcolors, Plotter
import sys
from math import pi


def loadBinFile(binfile,delimiter='\t',returnBinner=True):

    mat = np.loadtxt(binfile,delimiter=delimiter)

    left = mat[:,0]
    right = mat[:,1]
    try:
        center = mat[:,2]
    except:
        print("coreStats.py:loadBinFile says \"Third column absent in binfile. Using mean of left and right edges.\"")
        center = (left+right)/2.

    if returnBinner:
        bin_edges = left.copy()
        bin_edges = np.append(bin_edges,right[-1])
        return coreBinner(bin_edges)
    else:
        return left,right,center

class smartCls:

    def __init__(self,filename,ellrange=None,verbose=False):#colnum=-1,ellrange=None,norm="none",transpower=[0.,1.]):

        self.verbose = verbose
        #Do an initial pass through the file
        leng = -1
        maybe_ells = []
        self._fname = filename
        i=0
        with open(filename,'rb') as f:
            for line in f:

                row = line.split()
                assert(len(row)==leng or leng==-1), "ERROR: Not a column file."
                leng = len(row)
                if leng>1: maybe_ells.append(float(row[0]))
                i+=1
        #print leng
        #print i

        if not(ellrange==None):
            self.ells = ellrange
            
        elif leng>1. and min(maybe_ells)>1. and max(maybe_ells)<20000.:
            self.ells = maybe_ells
            if self.verbose: print("Found an ell column. Using it.")
        else:
            self.ells = np.arange(2.,i+2.,1.)
            print("Warning: no ell column detected. Assuming ell range is 2 to ~number of rows.")    
            #print "First column is ells"

        #if colnum>-1: return self.getCol(colnum,norm=norm,transpower=[0.,1.])
            
    def getCol(self,colnum=0,norm="none",transpower=[0.,1.]):

        col=[]
        i=0
        with open(self._fname,'rb') as f:
            for line in f:
                row = line.split()
                l = self.ells[i]
                    
                if norm=="none":
                    m = 1.
                elif norm=="lsq":
                    m = l*(l+1.)/2./pi
                    
                else:
                    print("ERROR: unrecognized norm factor")
                    sys.exit(1)


                p = transpower[1]*(l**(transpower[0]))    
                #print m
                #print row
                #print colnum
                #print row[colnum]
                col.append(float(row[colnum])*p/m)

                i+=1
                
        #print i
        return col
           

def validateMapType(mapXYType):
    assert not(re.search('[^TEB]', mapXYType)) and (len(mapXYType)==2), \
      bcolors.FAIL+"\""+mapXYType+"\" is an invalid map type. XY must be a two" + \
      " letter combination of T, E and B. e.g TT or TE."+bcolors.ENDC



def makeTemplate(l,Fl,mod,Nx,Ny,debug=False):
    """
    Given 1d function Fl of l, creates the 2d version
    of Fl on 2d k-space defined by mod
    """

    FlSpline = splrep(l,Fl,k=3) 
    ll = np.ravel(mod)
    kk = (splev(ll,FlSpline))


    template = np.reshape(kk,[Ny,Nx])

    
    if debug:
        print(kk)
        myFig = Plotter("$l$","$F_l$",scaleX="log",scaleY="log")
        #myFig.add(l,Fl)
        myFig.add(ll,kk)
        myFig.done(fileName="output/interp.png")
        plotme([mod],saveFile="output/mod.png",axoff=True,clbar=False)
        plotme([template],saveFile="output/temp.png",axoff=True,clbar=False)
        plotme([np.log(template)],saveFile="output/logtemp.png",axoff=True,clbar=False)
        sys.exit()
    

    return template

def loadCls(fileName, colnum, lpad, lmax, factorout="none"):
    '''
    WARNING: For autospectra, CAMB returns negative values beyond a certain ell, which is unphysical,
    so make sure your lpad is set to be less than that ell. Always examine your CAMB output.
    
    Load Cls from a CAMB-output file from column number colnum (>0, zero is assumed to hold ells)
    
    Pad with zeros after lpad
    Return lists that go up to lmax (possibly padded with zeros after lpad)
    factorout options
    1. none, get Cls as they are
    2. ll1, divide by l(l+1)/2pi
    3. ll1sq, divide by (l(l+1)^2/2pi
    4. ll13o2, divide by (l(l+1))^(3/2)/2pi
    5. l4, divide by l^4
    6. l3, divide by l^3
    Return list of ells and list of Cls
    '''


    uCell=[]
    ell=[]

    lFi=open(fileName,'r')
    for line in lFi:

        columns   = line.split()
        if (float(columns[0])>=lpad):
            break
        ell.append(float(columns[0]))
        uCell.append(float(columns[colnum]))



    ell=np.array(ell)


    if factorout=="none":
        uCellR=np.array(uCell)
    elif factorout=="ll1":
        uCellR=np.array([c*2.*pi/l/(l+1.) for c,l in zip(uCell,ell)])
    elif factorout=="ll1sq":
        uCellR=np.array([c*2.*pi/l/l/(l+1.)/(l+1.) for c,l in zip(uCell,ell)])
    elif factorout=="ll13o2":
        uCellR=np.array([c*2.*pi/((l/(l+1.))**(3./2.)) for c,l in zip(uCell,ell)])
    elif factorout=="l4":
        uCellR=np.array([c*2./(l**4.) for c,l in zip(uCell,ell)])
    elif factorout=="l3":
        uCellR=np.array([c*2./(l**3.) for c,l in zip(uCell,ell)])
    else:
        print((bcolors.FAIL+"ERROR: Unrecognized argument ", factorout," for factorout."+bcolors.ENDC))
        sys.exit()



    lFi.close()

    k=int(ell.max())
    apell = list(range(k+1,lmax))
    lastval = uCellR[-1]


    apucl = np.array([lastval]*len(apell))
    ell = np.append(ell,apell)
    uCellR = np.append(uCellR,apucl)
    
    


    return (ell), (uCellR)



class coreBinner:
    '''
    * Takes data defined on x0 and produces values binned on x.
    * Assumes x0 is linearly spaced and continuous in a domain?
    * Assumes x is continuous in a subdomain of x0.
    * Should handle NaNs correctly.
    '''
    

    def __init__(self, bin_edges):

        self.updateBinEdges(bin_edges)


    def updateBinEdges(self,bin_edges):
        
        self.bin_edges = bin_edges
        self.numbins = len(bin_edges)-1


    def binned(self,x,y):


        # pretty sure this treats nans in y correctly, but should double-check!
        bin_means = binnedstat(x,y,bins=self.bin_edges,statistic=np.nanmean)[0]


        
        return bin_means

        

    def getBinCenters(self,mode="mean"):

        if mode=="mean":
            return (self.bin_edges[:-1]+self.bin_edges[1:])/2.
        else:
            raise ValueError


def loadBinFile(binfile,delimiter='\t',returnBinner=True):

    mat = np.loadtxt(binfile,delimiter=delimiter)

    left = mat[:,0]
    right = mat[:,1]
    try:
        center = mat[:,2]
    except:
        print("coreStats.py:loadBinFile says \"Third column absent in binfile. Using mean of left and right edges.\"")
        center = (left+right)/2.

    if returnBinner:
        bin_edges = left.copy()
        bin_edges = np.append(bin_edges,right[-1])
        return coreBinner(bin_edges)
    else:
        return left,right,center
