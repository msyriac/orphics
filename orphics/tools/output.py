import matplotlib.pyplot as plt

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
        self._ax.legend(handles, labels,loc=loc,prop={'size':labsize},numpoints=1,**kwargs)
           
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
