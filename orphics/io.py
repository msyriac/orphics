from __future__ import print_function
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os,sys,logging,time
from orphics import mpi
import contextlib
import itertools

try:
    dout_dir = os.environ['WWW']+"plots/"
except:
    dout_dir = "./"

class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


## LOGGING

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
    
### FILE I/O

def dict_from_section(config,section_name):
    try:
        del config._sections[section_name]['__name__']
    except:
        pass
    return dict([a, list_from_config(config,section_name,a)[0]] for a, x in list(config._sections[section_name].items()))


    
def mkdir(dirpath,comm=mpi.MPI.COMM_WORLD):
    comm.Barrier()
    if comm.Get_rank()==0: 
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
    
def save_cols(filename,tuple_of_vectors,**kwargs):
    tuple_of_vectors = np.asarray(tuple_of_vectors)
    save_mat = np.vstack(tuple_of_vectors).T
    np.savetxt(filename,save_mat,**kwargs)

### NAMING
    
def join_nums(nums):
    return "_".join([str(f) for f in nums])


    
### CONFIG FILES
    
def load_path_config(filename=None):
    if filename is not None:
        return config_from_file(filename)
    else:
        if os.path.exists('input/paths_local.ini'):
            return config_from_file('input/paths_local.ini')
        elif os.path.exists('input/paths.ini'):
            return config_from_file('input/paths.ini')
        else:
            raise IOError
    
    
def config_from_file(filename):
    assert os.path.isfile(filename) 
    from configparser import SafeConfigParser 
    Config = SafeConfigParser()
    Config.optionxform=str
    Config.read(filename)
    return Config
    
def bin_edges_from_config(Config,section):
    from orphics.tools.stats import npspace

    spacing = Config.get(section,"spacing")
    minim = Config.getfloat(section,"left_edge")
    maxim = Config.getfloat(section,"right_edge")
    num = Config.getint(section,"num_bins")
    return npspace(minim,maxim,num,scale=spacing)

def list_from_string(string):
    return [float(x) for x in string.split(',')]

def list_from_config(Config,section,name):
    return list_from_string(Config.get(section,name))


### PLOTTING

def blend(fg_file,bg_file,alpha,save_file=None,verbose=True):
    from PIL import Image
    foreground = Image.open(fg_file)
    background = Image.open(bg_file)
    blended = Image.blend(foreground, background, alpha=alpha)
    if save_file is not None:
        blended.save(save_file)
        if verbose: cprint("Saved blended image to "+ save_file,color="g")
    return blended
    

def hist(data,bins=10,save_file=None,verbose=True,**kwargs):
    plt.hist(data,bins=bins,**kwargs)
    if save_file is not None:
        plt.savefig(save_file)
        if verbose: cprint("Saved histogram plot to "+ save_file,color="g")
    else:
        plt.show()
        

def mollview(hp_map,filename=None,cmin=None,cmax=None,coord='C',verbose=True,return_projected_map=False,**kwargs):
    '''
    mollview plot for healpix wrapper
    '''
    import healpy as hp
    retimg = hp.mollview(hp_map,min=cmin,max=cmax,coord=coord,return_projected_map=return_projected_map,**kwargs)
    if filename is not None:
        plt.savefig(filename)
        if verbose: cprint("Saved healpix plot to "+ filename,color="g")
    if return_projected_map: return retimg

def plot_img(array,filename=None,verbose=True,ftsize=24,high_res=False,flip=True,down=None,crange=None,**kwargs):
    if flip: array = np.flipud(array)
    if high_res:
        high_res_plot_img(array,filename,verbose=verbose,down=down,crange=crange,**kwargs)
    else:
        pl = Plotter(ftsize=ftsize,xlabel="",ylabel="")
        pl.plot2d(array,**kwargs)
        pl.done(filename,verbose=verbose)


def high_res_plot_img(array,filename=None,down=None,verbose=True,overwrite=True,crange=None):
    if not(overwrite):
        if os.path.isfile(filename): return
    try:
        from enlib import enmap, enplot
    except:
        traceback.print_exc()
        cprint("Could not produce plot "+filename+". High resolution plotting requires enlib, which couldn't be imported. Continuing without plotting.",color='fail')
        return
        
        
    if (down is not None) and (down!=1):
        downmap = enmap.downgrade(enmap.enmap(array)[None], down)
    else:
        downmap = enmap.enmap(array)[None]
    img = enplot.draw_map_field(downmap,enplot.parse_args("-vvvg moo"),crange=crange)
    #img = enplot.draw_map_field(downmap,enplot.parse_args("--grid 1"),crange=crange)
    if filename is None:
        img.show()
    else:
        img.save(filename)
        if verbose: print(bcolors.OKGREEN+"Saved high-res plot to", filename+bcolors.ENDC)


class Plotter(object):
    '''
    Fast, easy, and pretty publication-quality plots
    '''

    def __init__(self,xlabel=None,ylabel=None,xscale="linear",yscale="linear",ftsize=24,thk=1,labsize=None,major_tick_size=5,minor_tick_size=3,**kwargs):

        matplotlib.rc('axes', linewidth=thk)
        matplotlib.rc('axes', labelcolor='k')
        self.thk = thk
        
        self._fig=plt.figure(**kwargs)
        self._ax=self._fig.add_subplot(1,1,1)


        # Some self-disciplining :)
        try:
            force_label = os.environ['FORCE_ORPHICS_LABEL']
            force_label = True if force_label.lower().strip() == "true" else False
        except:
            force_label = False

        if force_label:
            assert xlabel is not None, "Please provide an xlabel for your plot"
            assert ylabel is not None, "Please provide a ylabel for your plot"



        if xlabel!=None: self._ax.set_xlabel(xlabel,fontsize=ftsize)
        if ylabel!=None: self._ax.set_ylabel(ylabel,fontsize=ftsize)

        self._ax.set_xscale(xscale, nonposx='clip') 
        self._ax.set_yscale(yscale, nonposy='clip')


        if labsize is None: labsize=ftsize
        plt.tick_params(axis='both', which='major', labelsize=labsize,width=self.thk,size=major_tick_size)#,size=labsize)
        plt.tick_params(axis='both', which='minor', labelsize=labsize,size=minor_tick_size)#,size=labsize)


    def legend(self,loc='upper left',labsize=10,numpoints=1,**kwargs):

        handles, labels = self._ax.get_legend_handles_labels()
        legend = self._ax.legend(handles, labels,loc=loc,prop={'size':labsize},numpoints=numpoints,frameon = 1,**kwargs)

        return legend
           
    def add(self,x,y,**kwargs):

        return self._ax.plot(x,y,**kwargs)


    def hist(self,data,**kwargs):
        return self._ax.hist(data,**kwargs)
    
        
    def add_err(self,x,y,yerr,ls='none',**kwargs):

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
        self._ax.axvline(x=x,ls=ls,alpha=alpha,color=color,**kwargs)

    def done(self,filename=None,verbose=True,**kwargs):

        if filename is not None:
            plt.savefig(filename,bbox_inches='tight',**kwargs)
            if verbose: cprint("Saved plot to "+ filename,"g")
        else:
            plt.show()

        plt.close()
    




# CONSOLE I/O
def cprint(string,color=None,bold=False,uline=False):
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
    
