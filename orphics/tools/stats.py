from __future__ import print_function
import warnings
warnings.warn("WARNING: This module is deprecated. Most of its contents have moved to orphics.stats. If you do not find the function you require there, please raise an issue.")
from scipy.stats import norm,binned_statistic as binnedstat,chi2
from scipy.optimize import curve_fit as cfit
from orphics.tools.io import Plotter,printC
import numpy as np
import time

def npspace(minim,maxim,num,scale="lin"):
    if scale=="lin" or scale=="linear":
        return np.linspace(minim,maxim,num)
    elif scale=="log":
        return np.logspace(np.log10(minim),np.log10(maxim),num)


def cov2corr(cov):
    # slow and stupid!
    
    d = np.diag(cov)
    stddev = np.sqrt(d)
    corr = cov.copy()*0.
    for i in range(cov.shape[0]):
        for j in range(cov.shape[0]):
            corr[i,j] = cov[i,j]/stddev[i]/stddev[j]

    return corr

def fchisq(dataVector,siginv,theoryVector=0.,amp=1.):
    
    diff = dataVector - amp*theoryVector
    b = np.dot(siginv,diff)
    chisq = np.dot(diff,b)
    return chisq




def getAmplitudeLikelihood(mean,covmat,amplitudeRange,theory):
    '''
    Returns the likelihood of mean w.r.t. theory over the 
    range amplitudeRange given covmat, i.e., compares
    mean to amp*theory for amp in amplitudeTheory.

    '''
    
    if covmat.size==1:
        siginv = 1./covmat
    else:
        siginv = np.linalg.pinv(covmat)

    print(siginv)
    #width = amplitudeRange[1]-amplitudeRange[0]
    

    Likelihood = lambda x: np.exp(-0.5*fchisq(mean,siginv,theory,amp=x))
    Likes = np.array([Likelihood(x) for x in amplitudeRange])
    Likes = Likes / np.trapz(Likes,amplitudeRange,np.diff(amplitudeRange)) #(Likes.sum()*width) #normalize
    return Likes


def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print(('%r %2.2f sec' % \
              (method.__name__,te-ts)))
        return result

    return timed


def getStats(listOfBinned):
    '''
    Returns statistics given an array-like object
    each row of which is an independent sample
    and each column holds possibly correlated
    binned values.

    The statistics are returned as a dictionary with
    keys:
    1. mean
    2. cov
    3. covmean
    4. err
    5. errmean
    6. corr
    '''
    # untested!
    
    
    arr = np.asarray(listOfBinned)
    N = arr.shape[0]  
    ret = {}
    ret['mean'] = np.nanmean(arr,axis=0)
    ret['cov'] = np.cov(arr.transpose())
    ret['covmean'] = ret['cov'] / N
    if arr.shape[1]==1:
        ret['err'] = np.sqrt(ret['cov'])
    else:
        ret['err'] = np.sqrt(np.diagonal(ret['cov']))
    ret['errmean'] = ret['err'] / np.sqrt(N)

    # correlation matrix
    if arr.shape[1]==1:
        ret['corr'] = 1.
    else:

        # ???
        d = np.diag(ret['cov'])
        stddev = np.sqrt(d)
        ret['corr'] = ret['cov'] / stddev[:, None]
        ret['corr'] = ret['cov'] / stddev[None, :]
        np.clip(ret['corr'], -1, 1, out=ret['corr'])
    

        
    return ret
    
def bootstrapData(profs,Bmax=100000):
    # profs is a list of 1d arrays you want to bootstrap over
    print("Bootstrapping...")
    newprofs = []
    for i in range(Bmax):
        
        xn = np.random.choice(len(profs),size=(1,len(profs)))
        xn = list(xn.flatten())

        newprof = sum([profs[i] for i in xn])/len(profs)
        newprofs.append(newprof)


    return newprofs
                                                        
class bin1D:
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


        
        return (self.bin_edges[:-1]+self.bin_edges[1:])/2.,bin_means

        

    # def getBinCenters(self,mode="mean"):

    #     if mode=="mean":
    #         return (self.bin_edges[:-1]+self.bin_edges[1:])/2.
    #     else:
    #         raise ValueError


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


class bin2D(object):
    def __init__(self, modRMap, bin_edges):
        self.centers = (bin_edges[1:]+bin_edges[:-1])/2.
        self.digitized = np.digitize(np.ndarray.flatten(modRMap), bin_edges,right=True)
        self.bin_edges = bin_edges
    def bin(self,data2d):
        data = np.ndarray.flatten(data2d)
        return self.centers,np.array([np.nanmean(data[self.digitized == i]) for i in range(1, len(self.bin_edges))])
    


def binInAnnuli(data2d, modRMap, bin_edges):
    binner = bin2D(modRMap, bin_edges)
    return binner.bin(data2d)



class ClBox:

    
    def __init__(self,binner=None,numNans=8):

        self.numNans = numNans
        self.binner = binner


        self.datas = {}
        

    def addCls(self,Cls,key,label,covmat,binned=False):
        # unbinned Cls always start at ell=0
        # Cls[0:numNans] won't matter whatever value they have
        # numNans should be at least 2 for sensible values in the first bin
        # use covmat = None to add a theory curve
        

        self.datas[key]={}
            

        if not(binned):
            Cls[0:self.numNans]=np.nan
            Clbinned = self.binner.binned(list(range(len(Cls))),Cls)
            self.datas[key]['unbinned'] = Cls
        else:
            Clbinned = Cls    

        cov = self._validateCovmat(covmat)    
        self.datas[key]['label']=label
        self.datas[key]['binned'] = Clbinned
        self.datas[key]['covmat'] = cov
        self.datas[key]['isFit'] = False
        if cov is None:
            self.datas[key]['amp']=(1.,0.)
        else:    
            self.datas[key]['siginv'] = np.linalg.pinv(cov)
            
        
        

    def _validateCovmat(self,covmat):
        
        if  isinstance(covmat, str):
            if covmat[:-4]==".pkl":
                cov = pickle.load(open(covmat,'rb'))
            else:
                cov = np.loadtxt(covmat)    
        elif isinstance(covmat, np.ndarray) or covmat is None:
            cov = covmat
        else:
            raise ValueError
        
        # if not(cov is None):
        #     try:
        #         assert cov.shape==(self.binner.numbins,self.binner.numbins)
        #     except:
        #         print cov.shape
        #         print self.binner.numbins
        #         sys.exit(1)    

        return cov
        

    def getStats(self,keyData,keyTheory,auto=False,show=False,numbins=-1):

        dataVector = self.datas[keyData]['binned'][:numbins]
        dofs = len(dataVector) - 2
        if auto: dofs -= 1
        chisqNull = self.chisq(keyData)
        chisqTheory = self.chisq(keyData,keyTheory,numbins=numbins)

        stats = {}
        stats['reduced chisquare'] =  chisqTheory / dofs
        stats['pte'] = chi2.sf(chisqTheory,dofs)
        stats['null sig'] = np.sqrt(chisqNull)
        stats['theory sig'] = np.sqrt(chisqNull-chisqTheory)
        

        if show:
            printC("="*len(keyTheory),color='y')
            printC(keyTheory,color='y')
            printC('-'*len(keyTheory),color='y')
            printC("amplitude",color='b')
            bf,err = self.datas[keyTheory]['amp']
            printC('{0:.2f}'.format(bf)+"+-"+'{:04.2f}'.format(err),color='p')
            
            for key,val in list(stats.items()):
                printC(key,color='b')
                printC('{0:.2f}'.format(val),color='p')
            printC("="*len(keyTheory),color='y')

        
        return stats

    def chisq(self,keyData,keyTheory=None,amp=1.,numbins=-1):

        dataVector = self.datas[keyData]['binned'][:numbins]
        if keyTheory is None:
            theoryVector = 0.
        else:     
            theoryVector = self.datas[keyTheory]['binned'][:numbins]
        siginv = self.datas[keyData]['siginv'][:numbins,:numbins]
        
        diff = dataVector - amp*theoryVector
        b = np.dot(siginv,diff)
        chisq = np.dot(diff,b)
        return chisq

    
        
    def fit(self,keyData,keyTheory,amplitudeRange=np.arange(0.1,2.0,0.01),debug=False,numbins=-1):
        # evaluate likelihood on a 1d grid and fit to a gaussian
        # store fit as new theory curve

        width = amplitudeRange[1]-amplitudeRange[0]
        Likelihood = lambda x: np.exp(-0.5*self.chisq(keyData,keyTheory,amp=x,numbins=numbins))
        Likes = np.array([Likelihood(x) for x in amplitudeRange])
        Likes = Likes / (Likes.sum()*width) #normalize

        ampBest,ampErr = cfit(norm.pdf,amplitudeRange,Likes,p0=[1.0,0.5])[0]

        
        
        if debug:
            fitVals = np.array([norm.pdf(x,ampBest,ampErr) for x in amplitudeRange])
            pl = Plotter()
            pl.add(amplitudeRange,Likes,label="likes")
            pl.add(amplitudeRange,fitVals,label="fit")
            pl.legendOn()
            pl.done("output/debug_coreFit.png")

        fitKey = keyData+"_fitTo_"+keyTheory
        self.datas[fitKey] = {}
        self.datas[fitKey]['covmat'] = None
        self.datas[fitKey]['binned'] = self.datas[keyTheory]['binned']*ampBest
        #self.datas[fitKey]['unbinned'] = self.datas[keyTheory]['unbinned']*ampBest
        self.datas[fitKey]['label'] = keyData+" fit to "+keyTheory+" with amp "+'{0:.2f}'.format(ampBest)+"+-"+'{0:.2f}'.format(ampErr)
        self.datas[fitKey]['amp']=(ampBest,ampErr)
        self.datas[fitKey]['isFit'] = True

        return fitKey
        

    def chisqAuto(self,keyData,keyTheory=None,amp=1.,const=0.):

        dataVector = self.datas[keyData]['binned']
        if keyTheory is None:
            theoryVector = 0.
        else:     
            theoryVector = self.datas[keyTheory]['binned']
        siginv = self.datas[keyData]['siginv']


        diff = (dataVector-const) - amp*theoryVector
        b = np.dot(siginv,diff)
        chisq = np.dot(diff,b)
        
        return chisq        
        
        
    def fitAuto(self,keyData,keyTheory,amplitudeRange=np.arange(0.1,2.0,0.01),constRange=np.arange(0.1,2.0,0.01),debug=False,store=False):
        # evaluate likelihood on a 2d grid and fit to a gaussian
        # store fit as new theory curve

        width = amplitudeRange[1]-amplitudeRange[0]
        height = constRange[1]-constRange[0]
        Likelihood = lambda x,y: np.exp(-0.5*self.chisqAuto(keyData,keyTheory,amp=x,const=y))
        #Likelihood = lambda x,y: -0.5*self.chisqAuto(keyData,keyTheory,amp=x,const=y)

        Likes = np.array([[Likelihood(x,y) for x in amplitudeRange] for y in constRange])

        ampLike = np.sum(Likes,axis=0)    
        constLike = np.sum(Likes,axis=1)

        ampLike = ampLike / (ampLike.sum()*width) #normalize
        constLike = constLike / (constLike.sum()*height) #normalize
                

        ampBest,ampErr = cfit(norm.pdf,amplitudeRange,ampLike,p0=[amplitudeRange.mean(),0.1*amplitudeRange.mean()])[0]
        constBest,constErr = cfit(norm.pdf,constRange,constLike,p0=[constRange.mean(),0.1*constRange.mean()])[0]


        if debug:
            pl = Plotter()
            pl.plot2d(Likes)
            pl.done("output/like2d.png")
                        
            pl = Plotter()
            fitVals = np.array([norm.pdf(x,ampBest,ampErr) for x in amplitudeRange])
            pl.add(amplitudeRange,ampLike,label="amplikes")
            pl.add(amplitudeRange,fitVals,label="fit")
            pl.legendOn()
            pl.done("output/amplike1d.png")

            pl = Plotter()
            fitVals = np.array([norm.pdf(x,constBest,constErr) for x in constRange])
            pl.add(constRange,constLike,label="constlikes")
            pl.add(constRange,fitVals,label="fit")
            pl.legendOn()
            pl.done("output/constlike1d.png")

            #sys.exit()
            
        if not(store):
            return constBest,constErr
        else:
            
            self.datas[keyData]['binned'] -= constBest
            self.datas[keyData]['unbinned'] -= constBest
            
            fitKey = keyData+"_fitTo_"+keyTheory
            self.datas[fitKey] = {}
            self.datas[fitKey]['covmat'] = None
            self.datas[fitKey]['binned'] = self.datas[keyTheory]['binned']*ampBest
            self.datas[fitKey]['unbinned'] = self.datas[keyTheory]['unbinned']*ampBest
            self.datas[fitKey]['label'] = keyData+" fit to "+keyTheory+" with amp "+'{0:.2f}'.format(ampBest)+"+-"+'{0:.2f}'.format(ampErr)
            self.datas[fitKey]['amp']=(ampBest,ampErr)
            self.datas[fitKey]['const']=(constBest,constErr)
            self.datas[fitKey]['isFit'] = True
    
            return fitKey



        

    def plotCls(self,saveFile,keys=None,xlimits=None,ylimits=None,transform=True,showBinnedTheory=False,scaleX='linear',scaleY='linear'):

        nsigma = 2.
        
        binCenters = self.binner.getBinCenters()

        if transform:
            ylab = "$\ell C_{\ell}$"
            mult = binCenters
            multTh = 1.#binCenters*0.+1.
        else:
            ylab = "$C_{\ell}$"
            mult = binCenters*0.+1.
            multTh = 0.#binCenters*0.
            
        pl = Plotter(labelX="$\ell$",labelY=ylab,scaleX=scaleX,scaleY=scaleY)


        
        if keys is None: keys = list(self.datas.keys())
        for key in keys:

            dat = self.datas[key]

            if dat['covmat'] is None:
                #This is a theory curve
                ells = np.array(list(range(len(dat['unbinned']))))
                if dat['isFit']:
                    ls="--"
                    lw=1
                else:
                    ls="-"
                    lw=2
                    
                base_line, = pl.add(ells,(multTh*(ells-1)+1.)*dat['unbinned'],label=dat['label'],lw=lw,ls=ls)
                if dat['isFit']:
                    pl._ax.fill_between(ells,(multTh*(ells-1)+1.)*dat['unbinned']*(1.-nsigma*dat['amp'][1]/dat['amp'][0]),(multTh*(ells-1)+1.)*dat['unbinned']*(1.+nsigma*dat['amp'][1]/dat['amp'][0]),alpha=0.3, facecolor=base_line.get_color())
                    
                if showBinnedTheory:
                    pl.add(binCenters[:len(dat['binned'])],mult[:len(dat['binned'])]*dat['binned'],
                           ls='none',marker='x',mew=2,markersize=10,label=dat['label']+' binned')
                  
            else:
                errs = np.sqrt(np.diagonal(dat['covmat']))
                print((dat['label']))
                pl.addErr(binCenters[:len(dat['binned'])],mult[:len(dat['binned'])]*dat['binned'],mult[:len(dat['binned'])]*errs,label=dat['label'],marker='o',elinewidth=2,markersize=10,mew=2,)


        [i.set_linewidth(2.0) for i in list(pl._ax.spines.values())]
        pl._ax.tick_params(which='major',width=2)
        pl._ax.tick_params(which='minor',width=2)
        pl._ax.axhline(y=0.,ls='--')
    
        if not(xlimits is None):
            pl._ax.set_xlim(*xlimits)
        else:
            pl._ax.set_xlim(self.binner.bin_edges[0],self.binner.bin_edges[-1])    
        if not(ylimits is None): pl._ax.set_ylim(*ylimits)
        pl.legendOn(loc='lower left',labsize=10)
        pl.done(saveFile)
