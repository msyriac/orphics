# import matplotlib
# matplotlib.rcParams['mathtext.fontset'] = 'custom'
# matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
# matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
# matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
import matplotlib.pyplot as plt


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

    def __init__(self,labelX=None,labelY=None,scaleX="linear",scaleY="linear",ftsize=24,**kwargs):

        self._fig=plt.figure(**kwargs)
        self._ax=self._fig.add_subplot(1,1,1)

        ax = self._ax

        if labelX!=None: ax.set_xlabel(labelX,fontsize=ftsize)
        if labelY!=None: ax.set_ylabel(labelY,fontsize=ftsize)

        ax.set_xscale(scaleX, nonposx='clip') 
        ax.set_yscale(scaleY, nonposy='clip')

    def legendOn(self,loc='upper left',labsize=18,**kwargs):
        handles, labels = self._ax.get_legend_handles_labels()
        legend = self._ax.legend(handles, labels,loc=loc,prop={'size':labsize},numpoints=1,frameon = 1,**kwargs)

        return legend
           
    def add(self,x,y,**kwargs):

        return self._ax.plot(x,y,**kwargs)
        
    def addErr(self,x,y,yerr,ls='none',**kwargs):

        self._ax.errorbar(x,y,yerr=yerr,ls=ls,**kwargs)

    def plot2d(self,data,lim=None,levels=None,clip=0,clbar=True,cm=None,label=None,labsize=18,extent=None,ticksize=12,**kwargs):

        Nx=data.shape[0]
        Ny=data.shape[1]
        arr=data[clip:Nx-clip,clip:Ny-clip]

        if lim==None:
            limmin=None
        else:
            limmin=-lim

        img = self._ax.imshow(arr,interpolation="none",vmin=limmin,vmax=lim,cmap=cm,extent=extent,**kwargs)

        if levels!=None:
           self._ax.contour(arr,levels=levels,extent=extent,origin="upper",colors=['black','black'],linestyles=['--','-'])

        
        if clbar:
            cbar = self._fig.colorbar(img)
            for t in cbar.ax.get_yticklabels():
                t.set_fontsize(ticksize)
            if label!=None:
                cbar.set_label(label,size=labsize)#,rotation=0)



            

                                

    def done(self,fileName="output/default.png"):

        plt.savefig(fileName,bbox_inches='tight')

        print bcolors.OKGREEN+"Saved plot to", fileName+bcolors.ENDC
        plt.close()
