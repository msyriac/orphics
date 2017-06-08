import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os.path

import contextlib
import sys

class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


def highResPlot2d(array,outPath,down=None,verbose=True,overwrite=True):
    if not(overwrite):
        if os.path.isfile(outPath): return
    try:
        from enlib import enmap, enplot
    except:
        printC("Could not produce plot "+outPath+". High resolution plotting requires enlib, which couldn't be imported. Continuing without plotting.",color='fail')
        return
        
        
    if (down is not None) and (down!=1):
        downmap = enmap.downgrade(enmap.enmap(array)[None], down)
    else:
        downmap = enmap.enmap(array)[None]
    img = enplot.draw_map_field(downmap,enplot.parse_args("-vvvg moo"))
    img.save(outPath)
    if verbose: print bcolors.OKGREEN+"Saved high-res plot to", outPath+bcolors.ENDC

    
def quickPlot2d(array,outPath,**kwargs):
    pl = Plotter()
    pl.plot2d(array,**kwargs)
    pl.done(outPath)


def getLensParams(Config,section):
    import numpy as np
    def setDefault(Config,section,name,default=0,min=0):
        try:
            val = Config.getfloat(section,name)
            assert val>=min
        except:
            val = default
        return val

    try:
        beamFile = Config.get(section,'beamFile')
        assert beamFile.strip()!=''
        beamArcmin = None
    except:
        beamFile = None
        beamArcmin = Config.getfloat(section,'beamArcmin')
        assert beamArcmin>0.

    try:
        fgFile = Config.get(section,'fgFile')
        assert beamFile.strip()!=''
    except:
        fgFile = None

    noiseT = Config.getfloat(section,'noiseT')
    noiseP = setDefault(Config,section,'noiseP',np.sqrt(2.)*noiseT,0)

    tellmin = Config.getint(section,'tellmin')    
    tellmax = Config.getint(section,'tellmax')    
    pellmin = Config.getint(section,'pellmin')    
    pellmax = Config.getint(section,'pellmax')

    
    lxcutT = int(setDefault(Config,section,'lxcutT',0,0))
    lycutT = int(setDefault(Config,section,'lycutT',0,0))
    lxcutP = int(setDefault(Config,section,'lxcutP',0,0))
    lycutP = int(setDefault(Config,section,'lycutP',0,0))

    lkneeT = setDefault(Config,section,'lkneeT',0,0)
    alphaT = setDefault(Config,section,'alphaT',0,-np.inf)
    lkneeP = setDefault(Config,section,'lkneeP',0,0)
    alphaP = setDefault(Config,section,'alphaP',0,-np.inf)

    

    return beamArcmin,beamFile,fgFile,noiseT,noiseP,tellmin,tellmax,pellmin,pellmax,lxcutT,lycutT,lxcutP,lycutP,lkneeT,alphaT,lkneeP,alphaP

def getListFromConfigSection(Config,section_name,key_root,start_index=0):

    i = start_index
    retList = []
    while True:
        key_name = key_root+"("+str(i)+")"
        try:
            val = Config._sections[section_name][key_name]
            retList.append(val)
        except KeyError:
            break
        i+=1

    return retList

def dictOfListsFromSection(config,sectionName):
    del config._sections[sectionName]['__name__']
    return dict([a, listFromConfig(config,sectionName,a)] for a, x in config._sections[sectionName].iteritems())

def dictFromSection(config,sectionName):
    try:
        del config._sections[sectionName]['__name__']
    except:
        pass
    return dict([a, listFromConfig(config,sectionName,a)[0]] for a, x in config._sections[sectionName].iteritems())


def listFromConfig(Config,section,name):
    return [float(x) for x in Config.get(section,name).split(',')]


def getFileNameString(listOfNames,listOfVals):
    fullstr = ""
    for name,val in zip(listOfNames,listOfVals):
        fullstr += "_"+name+"_"+str(val)

    return fullstr


def printC(string,color=None,bold=False,uline=False):
    if not(isinstance(string,str)):
        string = str(string)
    x=""
    if bold:
        x+=bcolors.BOLD
    if uline:
        x+=bcolors.UNDERLINE

    color = color.lower()    
    if color in ['b','blue']:
        x+=bcolors.OKBLUE
    elif color in ['r','red','f','fail']:
        x+=bcolors.FAIL
    elif color in ['g','green','ok']:
        x+=bcolors.OKGREEN
    elif color in ['y','yellow','w','warning']:
        x+=bcolors.WARNING
    elif color in ['p','purple','h','header']:
        x+=bcolors.HEADER
              

    

    print x+string+bcolors.ENDC


class bcolors:
    '''
    Colored output for print commands
    '''
    
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Plotter:
    '''
    Fast, easy, and pretty publication-quality plots
    '''

    def __init__(self,labelX=None,labelY=None,scaleX="linear",scaleY="linear",ftsize=24,thk=1,**kwargs):

        matplotlib.rc('axes', linewidth=thk)
        matplotlib.rc('axes', labelcolor='k')
        self.thk = thk
        
        self._fig=plt.figure(**kwargs)
        self._ax=self._fig.add_subplot(1,1,1)

        ax = self._ax



        if labelX!=None: ax.set_xlabel(labelX,fontsize=ftsize)
        if labelY!=None: ax.set_ylabel(labelY,fontsize=ftsize)

        ax.set_xscale(scaleX, nonposx='clip') 
        ax.set_yscale(scaleY, nonposy='clip')

    def legendOn(self,loc='upper left',labsize=18,**kwargs):
        plt.tick_params(size=labsize,width=self.thk,labelsize = labsize)

        handles, labels = self._ax.get_legend_handles_labels()
        legend = self._ax.legend(handles, labels,loc=loc,prop={'size':labsize},numpoints=1,frameon = 1,**kwargs)

        return legend
           
    def add(self,x,y,**kwargs):

        return self._ax.plot(x,y,**kwargs)
        
    def addErr(self,x,y,yerr,ls='none',**kwargs):

        self._ax.errorbar(x,y,yerr=yerr,ls=ls,**kwargs)

    def plot2d(self,data,lim=None,levels=None,clip=0,clbar=True,cm=None,label=None,labsize=18,extent=None,ticksize=12,**kwargs):
        '''
        For an array passed in as [j,i]
        Displays j along y and i along x , so (y,x)
        With the origin at upper left
        '''

        Nx=data.shape[0]
        Ny=data.shape[1]
        arr=data[clip:Nx-clip,clip:Ny-clip]

        if type(lim) is list:
            limmin,limmax = lim
        elif lim==None:
            limmin=None
            limmax = None
        else:
            limmin=-lim
            limmax = lim

        img = self._ax.imshow(arr,interpolation="none",vmin=limmin,vmax=limmax,cmap=cm,extent=extent,**kwargs)

        if levels!=None:
           self._ax.contour(arr,levels=levels,extent=extent,origin="upper",colors=['black','black'],linestyles=['--','-'])

        
        if clbar:
            cbar = self._fig.colorbar(img)
            for t in cbar.ax.get_yticklabels():
                t.set_fontsize(ticksize)
            if label!=None:
                cbar.set_label(label,size=labsize)#,rotation=0)



            

                                

    def done(self,fileName="output/default.png",**kwargs):

        plt.savefig(fileName,bbox_inches='tight',**kwargs)

        print bcolors.OKGREEN+"Saved plot to", fileName+bcolors.ENDC
        plt.close()



class FisherPlots(object):
    def __init__(self,paramList,paramLatexList,fidDict):
        self.fishers = {}
        self.fidDict = fidDict
        self.paramList = paramList
        self.paramLatexList = paramLatexList
        xx = np.array(np.arange(360) / 180. * np.pi)
        self.circl = np.array([np.cos(xx),np.sin(xx)])


    def addFisher(self,setName,fisherMat,gaussOnly=False):
        self.fishers[setName] = (gaussOnly,fisherMat)

    def plot1d(self,paramName,frange,setNames,cols=itertools.repeat(None),lss=itertools.repeat(None),labels=itertools.repeat(None),saveFile="default.png",labsize=12,labloc='upper left',xmultiplier=1.,labelXSuffix=""):

        fval = self.fidDict[paramName]
        i = self.paramList.index(paramName)
        paramlabel = '$'+self.paramLatexList[i]+'$' 

        pl = Plotter()
        errs = {}
        hasLabels = False
        for setName,col,ls,lab in zip(setNames,cols,lss,labels):
            if lab is not None: hasLabels = True
            gaussOnly, fisher = self.fishers[setName]
            if gaussOnly:
                ErrSigmaSq = fisher**2.
            else:
                Finv = np.linalg.inv(fisher)
                ErrSigmaSq = Finv[i,i]
            gaussFunc = lambda x: np.exp(-(x-fval)**2./2./ErrSigmaSq)
            pl.add(frange*xmultiplier,gaussFunc(frange),color=col,ls=ls,label=lab,lw=2)

        pl._ax.tick_params(size=14,width=3,labelsize = 16)
        pl._ax.set_xlabel(paramlabel+labelXSuffix,fontsize=24,weight='bold')
        if hasLabels: pl.legendOn(labsize=labsize,loc=labloc)
        pl.done(saveFile)
        
    def plotPair(self,paramXYPair,setNames,cols=itertools.repeat(None),lss=itertools.repeat(None),labels=itertools.repeat(None),saveFile="default.png",levels=[2.],xlims=None,ylims=None,loc='center',**kwargs):
        paramX,paramY = paramXYPair

        xval = self.fidDict[paramX]
        yval = self.fidDict[paramY]
        i = self.paramList.index(paramX)
        j = self.paramList.index(paramY)

        thk = 3
        #xx = np.array(np.arange(360) / 180. * np.pi)
        circl = self.circl #np.array([np.cos(xx),np.sin(xx)])


        paramlabely = '$'+self.paramLatexList[j]+'$' 
        paramlabelx = '$'+self.paramLatexList[i]+'$'
        
        matplotlib.rc('axes', linewidth=thk)
        matplotlib.rc('axes', labelcolor='k')
        #plt.figure(figsize=(6,5.5))

        fig=plt.figure(**kwargs)
        ax = fig.add_subplot(1,1,1)

        plt.tick_params(size=14,width=thk,labelsize = 16)

        if cols is None: cols = itertools.repeat(None)

        for setName,col,ls,lab in zip(setNames,cols,lss,labels):
            gaussOnly, fisher = self.fishers[setName]
            Finv = np.linalg.inv(fisher)
            chi211 = Finv[i,i]
            chi222 = Finv[j,j]
            chi212 = Finv[i,j]
        
            chisq = np.array([[chi211,chi212],[chi212,chi222]])

            Lmat = np.linalg.cholesky(chisq)
            ansout = np.dot(1.52*Lmat,circl)
            ax.plot(ansout[0,:]+xval, ansout[1,:]+yval,linewidth=thk,color=col,ls=ls,label=lab)
        




        ax.set_ylabel(paramlabely,fontsize=24,weight='bold')
        ax.set_xlabel(paramlabelx,fontsize=24,weight='bold')

        if xlims is not None: ax.set_xlim(*xlims)
        if ylims is not None: ax.set_ylim(*ylims)
        
        
        labsize = 12
        #loc = 'upper right'
        #loc = 'center'
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(handles, labels,loc=loc,prop={'size':labsize},numpoints=1,frameon = 0,**kwargs)


        plt.savefig(saveFile, bbox_inches='tight',format='png')
        print bcolors.OKGREEN+"Saved plot to", saveFile+bcolors.ENDC


    def plotTri(self,paramList,setNames,cols=itertools.repeat(None),lss=itertools.repeat(None),labels=itertools.repeat(None),saveFile="default.png",levels=[2.],xlims=None,ylims=None,loc='upper right',centerMarker=True,**kwargs):

        circl = self.circl
        numpars = len(paramList)
        thk = 3

        matplotlib.rc('axes', linewidth=thk)
        fig=plt.figure(figsize=(4*numpars,4*numpars),**kwargs)
        
        if cols is None: cols = itertools.repeat(None)

        for setName,col,ls,lab in zip(setNames,cols,lss,labels):
            gaussOnly, fisher = self.fishers[setName]
            Finv = np.linalg.inv(fisher)
            for i in xrange(0,numpars):
                for j in xrange(i+1,numpars):
                    count = 1+(j-1)*(numpars-1) + i

                    paramX = paramList[i]
                    paramY = paramList[j]

                    p = self.paramList.index(paramX)
                    q = self.paramList.index(paramY)

                    chi211 = Finv[p,p]
                    chi222 = Finv[q,q]
                    chi212 = Finv[p,q]

                    # a sigma8 hack
                    if "S8" in paramX:
                        xval = 0.8
                        paramlabelx = '$\sigma_8(z_{'+paramX[3:]+'})$'
                    else:
                        xval = self.fidDict[paramX]
                        paramlabelx = '$'+self.paramLatexList[p]+'$'
                    if "S8" in paramY:
                        yval = 0.8
                        paramlabely = '$\sigma_8(z_{'+paramY[3:]+'})$'
                    else:
                        yval = self.fidDict[paramY]
                        paramlabely = '$'+self.paramLatexList[q]+'$' 

                    if paramX=="S8All": paramlabelx = '$\sigma_8$'
                    if paramY=="S8All": paramlabely = '$\sigma_8$'
                        
                    chisq = np.array([[chi211,chi212],[chi212,chi222]])
                    Lmat = np.linalg.cholesky(chisq)
                    ansout = np.dot(1.52*Lmat,circl)

                    
                    
                    ax = fig.add_subplot(numpars-1,numpars-1,count)
                    plt.tick_params(size=14,width=thk,labelsize = 16)
                    if centerMarker: ax.plot(xval,yval,'xk',mew=thk)
                    ax.plot(ansout[0,:]+xval,ansout[1,:]+yval,linewidth=thk,color=col,ls=ls,label=lab)
                    if (i==0):#(count ==1):
                        ax.set_ylabel(paramlabely, fontsize=32,weight='bold')
                    if (j == (numpars-1)):
                        ax.set_xlabel(paramlabelx, fontsize=32,weight='bold')

        
        labsize = 32
        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels,prop={'size':labsize},numpoints=1,frameon = 0,loc=loc, bbox_to_anchor = (-0.1,-0.1,1,1),bbox_transform = plt.gcf().transFigure,**kwargs) #

        plt.savefig(saveFile, bbox_inches='tight',format='png')
        print bcolors.OKGREEN+"Saved plot to", saveFile+bcolors.ENDC
                    

