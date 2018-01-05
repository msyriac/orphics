from __future__ import print_function
import warnings
warnings.warn("WARNING: This module is deprecated. Most of its contents have moved to orphics.io. If you do not find the function you require there, please raise an issue.")
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os.path
import traceback
import contextlib
import os,sys,time,logging

try:
    dout_dir = os.environ['WWW']+"plots/"
except:
    dout_dir = "."

def quickMapView(hpMap,saveLoc=None,min=None,max=None,transform='C',**kwargs):
    '''
    Input map in galactic is shown in equatorial
    '''
    import healpy as hp
    hp.mollview(hpMap,min=min,max=max,coord=transform,**kwargs)
    if saveLoc==None: saveLoc="output/debug.png"
    matplotlib.pyplot.savefig(saveLoc)

    print(bcolors.OKGREEN+"Saved healpix plot to", saveLoc+bcolors.ENDC)

def save_dict_to_hdf5(dic, filename):
    """
    ....
    """
    import h5py

    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)

def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    ....
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type'%type(item))

def load_dict_from_hdf5(filename):
    """
    ....
    """
    import h5py

    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    import h5py

    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans
    
def bin_edges_from_config(Config,section):
    from orphics.tools.stats import npspace

    spacing = Config.get(section,"spacing")
    minim = Config.getfloat(section,"left_edge")
    maxim = Config.getfloat(section,"right_edge")
    num = Config.getint(section,"num_bins")
    return npspace(minim,maxim,num,scale=spacing)
        

class LoggerWriter:
    def __init__(self, level):
        # self.level is really like using log.debug(message)
        # at least in my case
        self.level = level

    def write(self, message):
        # if statement reduces the amount of newlines that are
        # printed to the logger
        if message != '\n':
            self.level(message)

    def flush(self):
        # create a flush method so things can be flushed when
        # the system wants to. Not sure if simply 'printing'
        # sys.stderr is the correct way to do it, but it seemed
        # to work properly for me.
        self.level(sys.stderr)

def load_path_config():
    if os.path.exists('input/paths_local.ini'):
        return config_from_file('input/paths_local.ini')
    elif os.path.exists('input/paths.ini'):
        return config_from_file('input/paths.ini')
    else:
        raise IOError

def get_logger(logname)        :
    logging.basicConfig(filename=logname+str(time.time()*10)+".log",level=logging.DEBUG,format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',datefmt='%m-%d %H:%M',filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger('')
    sys.stdout = LoggerWriter(logger.debug)
    sys.stderr = LoggerWriter(logger.warning)
    return logger    
    
def mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        
def save_cols(filename,tuple_of_vectors,**kwargs):
    save_mat = np.vstack(tuple_of_vectors).T
    np.savetxt(filename,save_mat,**kwargs)

def get_none_or_int(Config,section,name):
    val = Config.get(section,name)
    if val.strip().lower()=="none":
        return None
    else:
        return int(val)

def config_from_file(filename):
    assert os.path.isfile(filename) 
    from configparser import SafeConfigParser 
    Config = SafeConfigParser()
    Config.optionxform=str
    Config.read(filename)
    return Config


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def read_ignore_error(file_loc):
    vals = []
    with open(file_loc) as f:
        for line in f:
            try:
                val = float(line.strip())
                vals.append(val)
            except:
                print("Ignoring line ", line.strip())
    return np.array(vals)

def list_to_fits_table(arr,col_names,file_name):

    from astropy.table import Table
    t = Table(arr,names=col_names)
    t.write(file_name,format="fits")

class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


def highResPlot2d(array,outPath,down=None,verbose=True,overwrite=True,crange=None):
    if not(overwrite):
        if os.path.isfile(outPath): return
    try:
        from enlib import enmap, enplot
    except:
        traceback.print_exc()
        printC("Could not produce plot "+outPath+". High resolution plotting requires enlib, which couldn't be imported. Continuing without plotting.",color='fail')
        return
        
        
    if (down is not None) and (down!=1):
        downmap = enmap.downgrade(enmap.enmap(array)[None], down)
    else:
        downmap = enmap.enmap(array)[None]
    img = enplot.draw_map_field(downmap,enplot.parse_args("-vvvg moo"),crange=crange)
    img.save(outPath)
    if verbose: print(bcolors.OKGREEN+"Saved high-res plot to", outPath+bcolors.ENDC)

    
def quickPlot2d(array,outPath,verbose=True,ftsize=24,**kwargs):
    pl = Plotter(ftsize=ftsize)
    pl.plot2d(array,**kwargs)
    pl.done(outPath,verbose=verbose)


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
    return dict([a, listFromConfig(config,sectionName,a)] for a, x in list(config._sections[sectionName].items()))

def dictFromSection(config,sectionName):
    try:
        del config._sections[sectionName]['__name__']
    except:
        pass
    return dict([a, listFromConfig(config,sectionName,a)[0]] for a, x in list(config._sections[sectionName].items()))


def listFromConfig(Config,section,name):
    return [float(x) for x in Config.get(section,name).split(',')]


def get_none_or_int(Config,section,name):
    val = Config.get(section,name)
    if val.strip().lower()=="none":
        return None
    else:
        return int(val)


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
              

    

    print(x+string+bcolors.ENDC)


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

class Plotter(object):
    '''
    Fast, easy, and pretty publication-quality plots
    '''

    def __init__(self,labelX=None,labelY=None,scaleX="linear",scaleY="linear",ftsize=24,thk=1,labsize=None,major_tick_size=5,minor_tick_size=3,**kwargs):

        matplotlib.rc('axes', linewidth=thk)
        matplotlib.rc('axes', labelcolor='k')
        self.thk = thk
        
        self._fig=plt.figure(**kwargs)
        self._ax=self._fig.add_subplot(1,1,1)

        



        if labelX!=None: self._ax.set_xlabel(labelX,fontsize=ftsize)
        if labelY!=None: self._ax.set_ylabel(labelY,fontsize=ftsize)

        self._ax.set_xscale(scaleX, nonposx='clip') 
        self._ax.set_yscale(scaleY, nonposy='clip')


        if labsize is None: labsize=ftsize
        plt.tick_params(axis='both', which='major', labelsize=labsize,width=self.thk,size=major_tick_size)#,size=labsize)
        plt.tick_params(axis='both', which='minor', labelsize=labsize,size=minor_tick_size)#,size=labsize)


    def legendOn(self,loc='upper left',labsize=10,**kwargs):

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



            

    def hline(self,y=0.,ls="--",alpha=0.5,color="k",**kwargs):
        self._ax.axhline(y=y,ls=ls,alpha=alpha,color=color,**kwargs)
        
    def vline(self,x=0.,ls="--",alpha=0.5,color="k",**kwargs):
        self._ax.axhline(x=x,ls=ls,alpha=alpha,color=color,**kwargs)

    def done(self,fileName="output/default.png",verbose=True,**kwargs):

        plt.savefig(fileName,bbox_inches='tight',**kwargs)

        if verbose: print(bcolors.OKGREEN+"Saved plot to", fileName+bcolors.ENDC)
        plt.close()



class FisherPlots(object):
    def __init__(self):
        self.fishers = {}
        self.fidDicts = {}
        self.paramLists = {}
        self.paramLatexLists = {}
        xx = np.array(np.arange(360) / 180. * np.pi)
        self.circl = np.array([np.cos(xx),np.sin(xx)])

    def addSection(self,section,paramList,paramLatexList,fidDict):
        self.fishers[section] = {}
        self.fidDicts[section] = fidDict
        self.paramLists[section] = paramList
        self.paramLatexLists[section] = paramLatexList

    def addFisher(self,section,setName,fisherMat,gaussOnly=False):
        self.fishers[section][setName] = (gaussOnly,fisherMat)

    def plot1d(self,section,paramName,frange,setNames,cols=itertools.repeat(None),lss=itertools.repeat(None),labels=itertools.repeat(None),saveFile="default.png",labsize=12,labloc='upper left',xmultiplier=1.,labelXSuffix=""):

        fval = self.fidDicts[section][paramName]
        i = self.paramLists[section].index(paramName)
        paramlabel = '$'+self.paramLatexLists[section][i]+'$' 

        pl = Plotter()
        errs = {}
        hasLabels = False
        for setName,col,ls,lab in zip(setNames,cols,lss,labels):
            if lab is not None: hasLabels = True
            gaussOnly, fisher = self.fishers[section][setName]
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


    def startFig(self,**kwargs):
        self.fig = plt.figure(**kwargs)
        self.ax = self.fig.add_subplot(1,1,1)

    def plotPair(self,section,paramXYPair,setNames,cols=itertools.repeat(None),lss=itertools.repeat(None),labels=itertools.repeat(None),levels=[2.],xlims=None,ylims=None,loc='center',alphas=None,**kwargs):
        if alphas is None: alphas = [1]*len(setNames)
        paramX,paramY = paramXYPair

        xval = self.fidDicts[section][paramX]
        yval = self.fidDicts[section][paramY]
        i = self.paramLists[section].index(paramX)
        j = self.paramLists[section].index(paramY)

        thk = 3
        #xx = np.array(np.arange(360) / 180. * np.pi)
        circl = self.circl #np.array([np.cos(xx),np.sin(xx)])


        paramlabely = '$'+self.paramLatexLists[section][j]+'$' 
        paramlabelx = '$'+self.paramLatexLists[section][i]+'$'
        
        matplotlib.rc('axes', linewidth=thk)
        matplotlib.rc('axes', labelcolor='k')
        #plt.figure(figsize=(6,5.5))

        fig=self.fig #
        ax = self.ax 

        plt.tick_params(size=14,width=thk,labelsize = 16)

        if cols is None: cols = itertools.repeat(None)

        for setName,col,ls,lab,alpha in zip(setNames,cols,lss,labels,alphas):
            gaussOnly, fisher = self.fishers[section][setName]
            Finv = np.linalg.inv(fisher)
            chi211 = Finv[i,i]
            chi222 = Finv[j,j]
            chi212 = Finv[i,j]
        
            chisq = np.array([[chi211,chi212],[chi212,chi222]])

            Lmat = np.linalg.cholesky(chisq)
            ansout = np.dot(1.52*Lmat,circl)
            ax.plot(ansout[0,:]+xval, ansout[1,:]+yval,linewidth=thk,color=col,ls=ls,label=lab,alpha=alpha)
        




        ax.set_ylabel(paramlabely,fontsize=24,weight='bold')
        ax.set_xlabel(paramlabelx,fontsize=24,weight='bold')

        if xlims is not None: ax.set_xlim(*xlims)
        if ylims is not None: ax.set_ylim(*ylims)
        
        
        labsize = 12
        #loc = 'upper right'
        #loc = 'center'
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(handles, labels,loc=loc,prop={'size':labsize},numpoints=1,frameon = 0,**kwargs)

    def done(self,saveFile):
        plt.savefig(saveFile, bbox_inches='tight',format='png')
        print(bcolors.OKGREEN+"Saved plot to", saveFile+bcolors.ENDC)


    def plotTri(self,section,paramList,setNames,cols=itertools.repeat(None),lss=itertools.repeat(None),labels=itertools.repeat(None),saveFile="default.png",levels=[2.],xlims=None,ylims=None,loc='upper right',centerMarker=True,TwoSig=False,**kwargs):

        circl = self.circl
        numpars = len(paramList)
        thk = 3

        matplotlib.rc('axes', linewidth=thk)
        fig=plt.figure(figsize=(4*numpars,4*numpars),**kwargs)
        
        if cols is None: cols = itertools.repeat(None)

        for setName,col,ls,lab in zip(setNames,cols,lss,labels):
            gaussOnly, fisher = self.fishers[section][setName]
            Finv = np.linalg.inv(fisher)
            for i in range(0,numpars):
                for j in range(i+1,numpars):
                    count = 1+(j-1)*(numpars-1) + i

                    paramX = paramList[i]
                    paramY = paramList[j]

                    p = self.paramLists[section].index(paramX)
                    q = self.paramLists[section].index(paramY)

                    chi211 = Finv[p,p]
                    chi222 = Finv[q,q]
                    chi212 = Finv[p,q]

                    # a sigma8 hack
                    if "S8" in paramX:
                        xval = 0.8
                        paramlabelx = '$\sigma_8(z_{'+paramX[3:]+'})$'
                    else:
                        xval = self.fidDicts[section][paramX]
                        paramlabelx = '$'+self.paramLatexLists[section][p]+'$'
                    if "S8" in paramY:
                        yval = 0.8
                        paramlabely = '$\sigma_8(z_{'+paramY[3:]+'})$'
                    else:
                        yval = self.fidDicts[section][paramY]
                        paramlabely = '$'+self.paramLatexLists[section][q]+'$' 

                    if paramX=="S8All": paramlabelx = '$\sigma_8$'
                    if paramY=="S8All": paramlabely = '$\sigma_8$'
                        
                    chisq = np.array([[chi211,chi212],[chi212,chi222]])
                    Lmat = np.linalg.cholesky(chisq)
                    ansout = np.dot(1.52*Lmat,circl)
                    ansout2 = np.dot(2.0*1.52*Lmat,circl)
                    
                    
                    ax = fig.add_subplot(numpars-1,numpars-1,count)
                    plt.tick_params(size=14,width=thk,labelsize = 11)
                    if centerMarker: ax.plot(xval,yval,'xk',mew=thk)
                    ax.plot(ansout[0,:]+xval,ansout[1,:]+yval,linewidth=thk,color=col,ls=ls,label=lab)
                    if TwoSig:
                        ax.plot(ansout2[0,:]+xval,ansout2[1,:]+yval,linewidth=thk,color=col,ls=ls)
                    if (i==0):#(count ==1):
                        ax.set_ylabel(paramlabely, fontsize=32,weight='bold')
                    if (j == (numpars-1)):
                        ax.set_xlabel(paramlabelx, fontsize=32,weight='bold')
                    
        
        labsize = 48
        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels,prop={'size':labsize},numpoints=1,frameon = 0,loc=loc, bbox_to_anchor = (-0.1,-0.1,1,1),bbox_transform = plt.gcf().transFigure,**kwargs) #

        plt.savefig(saveFile, bbox_inches='tight',format='png')
        print(bcolors.OKGREEN+"Saved plot to", saveFile+bcolors.ENDC)
