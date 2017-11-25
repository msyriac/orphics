from __future__ import print_function
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
#import itertools
import traceback
#import contextlib
import os,sys,time,logging
#import h5py

try:
    dout_dir = os.environ['WWW']+"plots/"
except:
    dout_dir = "."

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def load_path_config():
    if os.path.exists('input/paths_local.ini'):
        return config_from_file('input/paths_local.ini')
    elif os.path.exists('input/paths.ini'):
        return config_from_file('input/paths.ini')
    else:
        raise IOError
    
def save_cols(filename,tuple_of_vectors,**kwargs):
    tuple_of_vectors = np.asarray(tuple_of_vectors)
    save_mat = np.vstack(tuple_of_vectors).T
    np.savetxt(filename,save_mat,**kwargs)

def config_from_file(filename):
    assert os.path.isfile(filename) 
    from configparser import SafeConfigParser 
    Config = SafeConfigParser()
    Config.optionxform=str
    Config.read(filename)
    return Config
    

def mollview(hp_map,filename=None,cmin=None,cmax=None,coord='C',verbose=True,return_projected_map=False,**kwargs):
    '''
    mollview plot for healpix wrapper
    '''
    import healpy as hp
    retimg = hp.mollview(hp_map,min=cmin,max=cmax,coord=coord,return_projected_map=return_projected_map,**kwargs)
    if filename is not None:
        matplotlib.pyplot.savefig(filename)
        if verbose: cprint("Saved healpix plot to "+ filename,color="g")
    if return_projected_map: return retimg

def plot_img(array,filename=None,verbose=True,ftsize=24,high_res=False,**kwargs):
    if high_res:
        high_res_plot_img(array,filename,verbose=verbose,**kwargs)
    else:
        pl = Plotter(ftsize=ftsize)
        pl.plot2d(array,**kwargs)
        pl.done(filename,verbose=verbose)


def high_res_plot_img(array,filename,down=None,verbose=True,overwrite=True,crange=None):
    if not(overwrite):
        if os.path.isfile(filename): return
    try:
        from enlib import enmap, enplot
    except:
        traceback.print_exc()
        printC("Could not produce plot "+filename+". High resolution plotting requires enlib, which couldn't be imported. Continuing without plotting.",color='fail')
        return
        
        
    if (down is not None) and (down!=1):
        downmap = enmap.downgrade(enmap.enmap(array)[None], down)
    else:
        downmap = enmap.enmap(array)[None]
    img = enplot.draw_map_field(downmap,enplot.parse_args("-vvvg moo"),crange=crange)
    img.save(filename)
    if verbose: print(bcolors.OKGREEN+"Saved high-res plot to", filename+bcolors.ENDC)

    
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

        



        if xlabel!=None: self._ax.set_xlabel(xlabel,fontsize=ftsize)
        if ylabel!=None: self._ax.set_ylabel(ylabel,fontsize=ftsize)

        self._ax.set_xscale(xscale, nonposx='clip') 
        self._ax.set_yscale(yscale, nonposy='clip')


        if labsize is None: labsize=ftsize
        plt.tick_params(axis='both', which='major', labelsize=labsize,width=self.thk,size=major_tick_size)#,size=labsize)
        plt.tick_params(axis='both', which='minor', labelsize=labsize,size=minor_tick_size)#,size=labsize)


    def legend(self,loc='upper left',labsize=10,**kwargs):

        handles, labels = self._ax.get_legend_handles_labels()
        legend = self._ax.legend(handles, labels,loc=loc,prop={'size':labsize},numpoints=1,frameon = 1,**kwargs)

        return legend
           
    def add(self,x,y,**kwargs):

        return self._ax.plot(x,y,**kwargs)
        
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
        self._ax.axhline(x=x,ls=ls,alpha=alpha,color=color,**kwargs)

    def done(self,filename=None,verbose=True,**kwargs):

        if filename is not None:
            plt.savefig(filename,bbox_inches='tight',**kwargs)
            if verbose: cprint("Saved plot to "+ filename,"g")
        else:
            plt.show()

        plt.close()
    


### CONFIG FILES

def bin_edges_from_config(Config,section):
    from orphics.tools.stats import npspace

    spacing = Config.get(section,"spacing")
    minim = Config.getfloat(section,"left_edge")
    maxim = Config.getfloat(section,"right_edge")
    num = Config.getint(section,"num_bins")
    return npspace(minim,maxim,num,scale=spacing)
