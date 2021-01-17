from __future__ import print_function
import numpy as np
import time, warnings
import itertools
import scipy
from scipy.stats import binned_statistic as binnedstat,chi2
from scipy.optimize import curve_fit
import itertools
from pyfisher import FisherMatrix

def eig_analyze(cmb2d,start=0,eigfunc=np.linalg.eigh,plot_file=None):
    es = eigfunc(cmb2d[start:,start:,...].T)[0]
    print(start,es.min(),np.any(es<0.))
    numw = range(np.prod(es.shape[:-1]))
    pl = io.Plotter(xlabel='n',ylabel='e',yscale='log')
    for ind in range(es.shape[-1]):
        pl.add(numw,np.sort(np.real(es[...,ind].ravel())))
        pl.add(numw,np.sort(np.imag(es[...,ind].ravel())),ls="--")
    pl.done(plot_file)


def get_sigma2(ells,cls,w0,delta_ells,fsky,ell0=0,alpha=1,w0p=None,ell0p=0,alphap=1,clxx=None,clyy=None):
    afact = ((ell0/ells)**(-alpha)) if ell0>1.e-3 else 0.*ells
    nlxx = (w0*np.pi/180./60.)**2 * afact
    if clxx is not None:
        afact = ((ell0p/ells)**(-alphap)) if ell0>1.e-3 else 0.*ells
        nlyy = (w0p*np.pi/180./60.)**2 * afact
        tclxx = clxx + nlxx
        tclyy = clyy + nlyy
        tcl2 = cls**2 + tclxx*tclyy
    else:
        assert clyy is None
        assert w0p is None
        tcl2 = 2 * (cls+nlxx)**2
    return tcl2/(2*ells+1)/fsky/delta_ells

def fit_cltt_power(ells,cls,cltt_func,w0,sigma2,ell0=0,alpha=1,fix_knee=False):
    """
    Fit binned power spectra to a linear model of the form
    A * C_ell + B * w0^2 (ell/ell0)^alpha + C * w0^2
    """
    # Cinv = np.diag(1./sigma2)
    # Cov = np.diag(sigma2)
    sw0 = w0 * np.pi / 180./ 60.
    if fix_knee:
        funcs = [lambda x: sw0**2]
    else:
        funcs = [lambda x: sw0**2, lambda x: sw0**2 * (ell0/x)**(-alpha) if ell0>1e-3 else sw0**2 ]
    bds = (0,np.inf)
    X,_ = curve_fit(lambda x,*args: sum([arg*f(x) for f,arg in zip(funcs,args) ]) , ells, (cls-cltt_func(ells)),
                  p0=[1] if fix_knee else [1,ell0],sigma=np.sqrt(sigma2),absolute_sigma=True,
                  bounds=bds)
    #X,_,_,_ = fit_linear_model(ells,(cls-cltt_func(ells)),Cov,funcs,Cinv=Cinv) # Linear models dont allow bounds
    return lambda x : cltt_func(x) + sum([ coeff*f(x)  for coeff,f in zip(X,funcs)])
        

def fit_linear_model(x,y,ycov,funcs,dofs=None,deproject=True,Cinv=None,Cy=None):
    """
    Given measurements with known uncertainties, this function fits those to a linear model:
    y = a0*funcs[0](x) + a1*funcs[1](x) + ...
    and returns the best fit coefficients a0,a1,... and their uncertainties as a covariance matrix
    """
    s = solve if deproject else np.linalg.solve
    C = ycov
    y = y[:,None] 
    A = np.zeros((y.size,len(funcs)))
    for i,func in enumerate(funcs):
        A[:,i] = func(x)
    CA = s(C,A) if Cinv is None else np.dot(Cinv,A)
    cov = np.linalg.inv(np.dot(A.T,CA))
    if Cy is None: Cy = s(C,y) if Cinv is None else np.dot(Cinv,y)
    b = np.dot(A.T,Cy)
    X = np.dot(cov,b)
    YAX = y - np.dot(A,X)
    CYAX = s(C,YAX) if Cinv is None else np.dot(Cinv,YAX)
    chisquare = np.dot(YAX.T,CYAX)
    dofs = len(x)-len(funcs)-1 if dofs is None else dofs
    pte = 1 - chi2.cdf(chisquare, dofs)    
    return X,cov,chisquare/dofs,pte
    
def fit_gauss(x,y,mu_guess=None,sigma_guess=None):
    ynorm = np.trapz(y,x)
    ynormalized = y/ynorm
    gaussian = lambda t,mu,sigma: np.exp(-(t-mu)**2./2./sigma**2.)/np.sqrt(2.*np.pi*sigma**2.)
    popt,pcov = curve_fit(gaussian,x,ynormalized,p0=[mu_guess,sigma_guess])#,bounds=([-np.inf,0],[np.inf,np.inf]))
    fit_mean = popt[0]
    fit_sigma = popt[1]
    return fit_mean,np.abs(fit_sigma),ynorm,ynormalized

    
class Solver(object):
    """
    Calculate Cinv . x
    """
    def __init__(self,C,u=None):
        """
        C is an (NxN) covariance matrix
        u is an (Nxk) template matrix for rank-k deprojection
        """
        N = C.shape[0]
        if u is None: u = np.ones((N,1))
        Cinvu = np.linalg.solve(C,u)
        self.precalc = np.dot(Cinvu,np.linalg.solve(np.dot(u.T,Cinvu),u.T))
        self.C = C
    def solve(self,x):
        Cinvx = np.linalg.solve(self.C,x)
        correction = np.dot(self.precalc,Cinvx)
        return Cinvx - correction
    
def solve(C,x,u=None):
    """
    Typically, you do not want to invert your covariance matrix C, but really just want
    Cinv . x , for some vector x.
    You can get that with np.linalg.solve(C,x).
    This function goes one step further and deprojects a common mode for the entire
    covariance matrix, which is often what is needed.
    """
    N = C.shape[0]
    s = Solver(C,u=u)
    return s.solve(x)


def alpha_from_confidence(c):
    """Returns the number of sigmas that corresponds to enclosure of c % of the
    probability density in a 2D gaussian.

    e.g. alpha_from_confidence(0.683) = 1.52
    """
    return np.sqrt(2.*np.log((1./(1.-c))))
    
def corner_plot(fishers,labels,fid_dict=None,params=None,confidence_level=0.683,show_1d=False,
                latex_dict=None,colors=itertools.repeat(None),lss=itertools.repeat(None),
                thk=2,center_marker=True,save_file=None,loc='upper right',labelsize=14,ticksize=2,lw=3,**kwargs):
    """Make a triangle/corner plot from Fisher matrices.
    Does not support multiple confidence levels. (Redundant under Gaussian approximation of Fisher)

    fishers -- list of stats.FisherMatrix objects
    labels -- labels corresponding to each Fisher matrix in fishers
    params -- By default (if None) uses all params in every Fisher matrix. If not None, uses only this list of params.
    fid_dict -- dictionary mapping parameter names to fiducial values to center ellipses on
    latex_dict -- dictionary mapping parameter names to LaTeX strings for axes labels. Defaults to parameter names.
    confidence_level -- fraction of probability density enclosed by ellipse
    colors -- list of colors corresponding to fishers
    lss -- list of line styles corresponding to fishers
    """

    from orphics import io
    import matplotlib
    import matplotlib.pyplot as plt
    if params is None:
        ps = [f.params for f in fishers]
        jparams = list(set([p for sub_ps in ps for p in sub_ps]))
    else:
        jparams = params
    numpars = len(jparams)
    alpha = alpha_from_confidence(confidence_level)
    xx = np.array(np.arange(360) / 180. * np.pi)
    circl = np.array([np.cos(xx),np.sin(xx)])
    
    fig=plt.figure(figsize=(2*(numpars+1),2*(numpars+1)),**kwargs) if show_1d else plt.figure(figsize=(2*numpars,2*numpars),**kwargs)
    startp = 0 if show_1d else 1

    if show_1d:
        sigmas = []
        for fish in fishers:
            sigmas.append(fish.sigmas())
    else:
        sigmas = itertools.repeat(None)
        
    
    for i in range(0,numpars):
        for j in range(i+startp,numpars):
            count = 1+(j)*(numpars) + (i+1) -1 if show_1d else 1+(j-1)*(numpars-1) + i
            paramX = jparams[i]
            paramY = jparams[j]
            ax = fig.add_subplot(numpars,numpars,count) if show_1d else fig.add_subplot(numpars-1,numpars-1,count)

            if i>0:
                ax.yaxis.set_visible(False)
            if j!=(numpars-1):
                ax.xaxis.set_visible(False)
            
            try:
                xval = fid_dict[paramX]
            except:
                xval = 0.
            try:
                yval = fid_dict[paramY]
            except:
                yval = 0.
            try:
                paramlabely = '$%s$' % latex_dict[paramY]
            except:
                paramlabely = '$%s$' % paramY
            try:
                paramlabelx = '$%s$' % latex_dict[paramX]
            except:
                paramlabelx = '$%s$' % paramX
            if center_marker:
                if i==j:
                    ax.axvline(x=xval,ls="--",color='k',alpha=0.5)
                else:
                    ax.plot(xval,yval,'xk',mew=thk)

            for fish,ls,col,lab,sig in zip(fishers,lss,colors,labels,sigmas):

                if i==j:
                    try:
                        s = sig[paramX]
                    except:
                        continue
                    nsigma = 5
                    xs = np.linspace(xval-nsigma*s,xval+nsigma*s,1000)
                    from scipy.stats import norm
                    g = norm.pdf(xs,loc=xval,scale=s)
                    ax.plot(xs,g/g.max())
                    continue
                
                
                try:
                    chisq = fish.marge_var_2param(paramX,paramY)
                except:
                    ax.plot([xval],[yval],linewidth=0,color=col,ls=ls,label=lab,alpha=0) # just to iterate the colors and ls
                    continue
                Lmat = np.linalg.cholesky(chisq)
                ansout = np.dot(alpha*Lmat,circl)
                ax.plot(ansout[0,:]+xval,ansout[1,:]+yval,linewidth=lw,color=col,ls=ls,label=lab)
            if (i==0):
                ax.set_ylabel(paramlabely, fontsize=labelsize,weight='bold')
            if (j == (numpars-1)):
                ax.set_xlabel(paramlabelx, fontsize=labelsize,weight='bold')
    
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(handles, labels,prop={'size':labelsize},numpoints=1,frameon = 0,loc=loc, bbox_to_anchor = (-0.1,-0.1,1,1),bbox_transform = plt.gcf().transFigure,**kwargs)

    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file, bbox_inches='tight')
        print(io.bcolors.OKGREEN+"Saved plot to", save_file+io.bcolors.ENDC)
        

class OQE(object):
    """Optimal Quadratic Estimator for likelihoods that
    are Gaussian in the model parameters.
    
    WARNING: This class has not been tested thoroughly.

    Given a fiducial covariance matrix for the data and derivatives
    of the covariance matrix w.r.t. each parameter of interest,
    this class precalculates things like the Fisher matrix when
    initialized.

    Subsequently, given a data vector, it returns the OQEstimate
    as a dictionary in the parameters. 
    """
    def __init__(self,fid_cov,dcov_dict,fid_params_dict,invert=False,deproject=True,templates=None):

        self.params = dcov_dict.keys()
        self.fids = fid_params_dict
        Nparams = len(self.params)
        self.invert = invert
        if invert:
            self.Cinv = self._inv(fid_cov)
            

        if templates is not None: assert deproject
        self.biases = {}
        self.ps = {}
        for param in self.params:
            if not(invert):
                if deproject:
                    solution = solve(fid_cov,dcov_dict[param],u=templates)
                else:
                    solution = np.linalg.solve(fid_cov,dcov_dict[param])
            self.ps[param] = np.dot(self.Cinv,dcov_dict[param]) if invert else solution.copy()
            self.biases[param] = np.trace(self.ps[param])

        self.Fisher = np.zeros((Nparams,Nparams))
        param_combs = itertools.combinations_with_replacement(self.params,2)
        for param1,param2 in param_combs:
            i = self.params.index(param1)
            j = self.params.index(param2)
            self.Fisher[i,j] = 0.5 * np.trace(np.dot(self.ps[param1],self.ps[param2]))
            if j!=i: self.Fisher[j,i] = self.Fisher[i,j]

        self.Finv = self._inv(self.Fisher)

        self.marg_errors = np.diagonal(self.Finv)**(1./2.)

        if not(invert):
            if deproject:
                self.s = Solver(fid_cov,u=templates)
                self.solver = lambda x: self.s.solve(x)
            else:
                self.solver = lambda x: np.linalg.solve(fid_cov,x)

    def sigma(self):
        return dict(zip(self.params,self.marg_errors.tolist()))

    def estimate(self,data):
        vec = []
        for param in self.params:
            cinvdat = np.dot(self.Cinv,data) if self.invert else self.solver(data)
            fcore = np.dot(np.dot(data.T,self.ps[param]),cinvdat)
            bsubbed = fcore - self.biases[param]
            assert bsubbed.size == 1
            vec.append(bsubbed)
        vec = np.asarray(vec)
        ans = 0.5 * np.dot(self.Finv,vec)
        ans = dict(zip(self.params,ans.tolist()))
        res = {}
        for param in self.params:
            res[param] = self.fids[param] + ans[param]
        return res

        
    def _inv(self,cov):
        # return np.linalg.pinv(cov)
        return np.linalg.inv(cov)
        # return scipy.linalg.pinv2(cov)



class OQESlim(object):
    def __init__(self,fid_cov,dcov_dict,fid_params_dict,templates=None):

        self.params = dcov_dict.keys()
        self.fids = fid_params_dict
        Nparams = len(self.params)
        self.biases = {}
        self.ps = {}
        for param in self.params:
            solution = solve(fid_cov,dcov_dict[param],u=templates)
            self.ps[param] = solution.copy()
            self.biases[param] = np.trace(self.ps[param])

        self.Fisher = np.zeros((Nparams,Nparams))
        param_combs = itertools.combinations_with_replacement(self.params,2)
        for param1,param2 in param_combs:
            i = self.params.index(param1)
            j = self.params.index(param2)
            self.Fisher[i,j] = 0.5 * np.trace(np.dot(self.ps[param1],self.ps[param2]))
            if j!=i: self.Fisher[j,i] = self.Fisher[i,j]

        self.Finv = np.linalg.inv(self.Fisher)
        self.marg_errors = np.diagonal(self.Finv)**(1./2.)

        self.s = Solver(fid_cov,u=templates)
        self.solver = lambda x: self.s.solve(x)

    def sigma(self):
        return dict(zip(self.params,self.marg_errors.tolist()))

    def estimate(self,data):
        vec = []
        for param in self.params:
            cinvdat = self.solver(data)
            fcore = np.dot(np.dot(data.T,self.ps[param]),cinvdat)
            bsubbed = fcore - self.biases[param]
            assert bsubbed.size == 1
            vec.append(bsubbed)
        vec = np.asarray(vec)
        ans = 0.5 * np.dot(self.Finv,vec)
        ans = dict(zip(self.params,ans.tolist()))
        res = {}
        for param in self.params:
            res[param] = self.fids[param] + ans[param]
        return res

        
class CinvUpdater(object):

    def __init__(self,cinvs,logdets,profile):
        self.cinvs = cinvs
        self.logdets = logdets

        u = profile.reshape((len(profile),1))
        v = u.copy()
        vT = v.T
        self.update_unnormalized = []
        self.det_unnormalized = []
        for Ainv in cinvs:
            self.update_unnormalized.append( np.dot(Ainv, np.dot(np.dot(u,vT), Ainv)) )
            self.det_unnormalized.append( np.dot(vT, np.dot(Ainv, u)) )

    def get_cinv(self,index,amplitude):
        
        det_update = 1.+(amplitude**2.)*self.det_unnormalized[index]
        cinv_updated = self.cinvs[index] - (amplitude**2.)*( self.update_unnormalized[index]/ det_update)
        return  cinv_updated , np.log(det_update)+self.logdets[index]

        

def eig_pow(C,exponent=-1,lim=1.e-8):
    e,v = np.linalg.eigh(C)
    emax = np.max(e)
    mask = e<emax*lim
    e[~mask] **= exponent
    e[mask]=0.
    return (v*e).dot(v.T)

def sm_update(Ainv, u, v=None):
    """Compute the value of (A + uv^T)^-1 given A^-1, u, and v. 
    Uses the Sherman-Morrison formula."""

    v = u.copy() if v is None else v
    u = u.reshape((len(u),1))
    v = v.reshape((len(v),1))
    vT = v.T
    
    ldot = np.dot(vT, np.dot(Ainv, u))
    assert ldot.size==1
    det_update = 1.+ldot.ravel()[0]

    ans = Ainv - (np.dot(Ainv, np.dot(np.dot(u,vT), Ainv)) / det_update)
    return ans, det_update

def cov2corr_legacy(cov):
    # slow and stupid! see cov2corr
    d = np.diag(cov)
    stddev = np.sqrt(d)
    corr = cov.copy()*0.
    for i in range(cov.shape[0]):
        for j in range(cov.shape[0]):
            corr[i,j] = cov[i,j]/stddev[i]/stddev[j]
    return corr

def cov2corr(mat):
    diags = np.diagonal(mat).T
    xdiags = diags[:,None,...]
    ydiags = diags[None,:,...]
    corr = mat/np.sqrt(xdiags*ydiags)
    return corr
    
def correlated_hybrid_matrix(data_covmat,theory_covmat=None,theory_corr=None,cap=True,cap_off=0.99):
    """
    Given a diagonal matrix data_covmat,
    and a theory matrix theory_covmat or its correlation matrix theory_corr,
    produce a hybrid non-diagonal matrix that has the same diagonals as the data matrix
    but has correlation coefficient given by theory.
    """
    if theory_corr is None:
        assert theory_covmat is not None
        theory_corr = cov2corr(theory_covmat)
    r = theory_corr

    def _cap(imat,cval,csel):
        imat[imat>1] = 1
        imat[imat<-1] = -1
        imat[csel][imat[csel]>cval] = cval
        imat[csel][imat[csel]>-cval] = -cval

    d = data_covmat.copy()
    sel = np.where(~np.eye(d.shape[0],dtype=bool))
    d[sel] = 1
    dcorr = 1./cov2corr(d)
    if cap: _cap(r,cap_off,sel)
    fcorr = dcorr * r
    d[sel] = fcorr[sel]
    return d


class Stats(object):
    """
    An MPI enabled helper container for
    1) 1d measurements whose statistics need to be calculated
    2) 2d cumulative stacks

    where different MPI cores may be calculating different number
    of 1d measurements or 2d stacks.
    """
    
    def __init__(self,comm=None,root=0,loopover=None,tag_start=333):
        """
        comm - MPI.COMM_WORLD object
        tag_start - MPI comm tags start at this integer
        """

        if comm is not None:
            self.comm = comm
        else:
            from orphics.mpi import fakeMpiComm
            self.comm = fakeMpiComm()

            
        self.rank = self.comm.Get_rank()
        self.numcores = self.comm.Get_size()
        self.columns = {}
            
        self.vectors = {}
        self.little_stack = {}
        self.little_stack_count = {}
        self.tag_start = tag_start
        self.root = root
        if loopover is None:
            self.loopover = list(range(root+1,self.numcores))
        else:
            self.loopover = loopover

    def add_to_stats(self,label,vector,exclude=False):
        """
        Append the 1d vector to a statistic named "label".
        Create a new one if it doesn't already exist.
        """
        assert label!='stats', "Sorry, 'stats' is a forbidden label."

        vector = np.asarray(vector)

        if np.iscomplexobj(vector):
            print("ERROR: stats on complex arrays not supported. Do the real and imaginary parts separately.")
            raise TypeError
        
        if not(label in list(self.vectors.keys())):
            self.vectors[label] = []
            self.columns[label] = vector.shape
        if not(exclude):
            self.vectors[label].append(vector)


    def add_to_stack(self,label,arr,exclude=False):
        """
        This is just an accumulator, it can't track statisitics.
        Add arr to a cumulative stack named "label". Could be 2d arrays.
        Create a new one if it doesn't already exist.
        """
        assert label!='stats', "Sorry, 'stats' is a forbidden label."

        if np.iscomplexobj(arr):
            print("ERROR: stacking of complex arrays not supported. Stack the real and imaginary parts separately.")
            raise TypeError
        if not(label in list(self.little_stack.keys())):
            self.little_stack[label] = arr*0.
            self.little_stack_count[label] = 0
        if not(exclude):
            self.little_stack[label] += arr
            self.little_stack_count[label] += 1


    def get_stacks(self,verbose=True):
        """
        Collect from all MPI cores and calculate stacks.
        """

        if self.rank in self.loopover:

            for k,label in enumerate(self.little_stack.keys()):
                self.comm.send(self.little_stack_count[label], dest=self.root, tag=self.tag_start*3000+k)
            
            for k,label in enumerate(self.little_stack.keys()):
                send_dat = np.array(self.little_stack[label]).astype(np.float64)
                self.comm.Send(send_dat, dest=self.root, tag=self.tag_start*10+k)

        elif self.rank==self.root:
            self.stacks = {}
            self.stack_count = {}

            for k,label in enumerate(self.little_stack.keys()):
                self.stack_count[label] = self.little_stack_count[label]
                for core in self.loopover: 
                    if verbose: print(f"{label} waiting for count from core ", core , " / ", self.numcores)
                    data = self.comm.recv(source=core, tag=self.tag_start*3000+k)
                    self.stack_count[label] += data

            
            for k,label in enumerate(self.little_stack.keys()):
                self.stacks[label] = self.little_stack[label]
            for core in self.loopover: 
                if verbose: print(f"{label} waiting for data from core ", core , " / ", self.numcores)
                for k,label in enumerate(self.little_stack.keys()):
                    expected_shape = self.little_stack[label].shape
                    data_vessel = np.empty(expected_shape, dtype=np.float64)
                    self.comm.Recv(data_vessel, source=core, tag=self.tag_start*10+k)
                    self.stacks[label] += data_vessel

                    
            for k,label in enumerate(self.little_stack.keys()):                
                self.stacks[label] /= self.stack_count[label]
                
    def get_stats(self,verbose=True,skip_stats=False):
        """
        Collect from all MPI cores and calculate statistics for
        1d measurements.
        """

        if self.rank in self.loopover:
            for k,label in enumerate(self.vectors.keys()):
                self.comm.send(np.array(self.vectors[label]).shape[0], dest=self.root, tag=self.tag_start*2000+k)

            for k,label in enumerate(self.vectors.keys()):
                send_dat = np.array(self.vectors[label]).astype(np.float64)
                self.comm.Send(send_dat, dest=self.root, tag=self.tag_start+k)

        else:
            self.stats = {}
            self.numobj = {}
            for k,label in enumerate(self.vectors.keys()):
                self.numobj[label] = []
                self.numobj[label].append(np.array(self.vectors[label]).shape[0])
                for core in self.loopover: #range(1,self.numcores):
                    if verbose: print(f"{label} waiting for size from core ", core , " / ", self.numcores)
                    data = self.comm.recv(source=core, tag=self.tag_start*2000+k)
                    self.numobj[label].append(data)

            
            for k,label in enumerate(self.vectors.keys()):
                self.vectors[label] = np.array(self.vectors[label])
            for core in self.loopover: #range(1,self.numcores):
                if verbose: print(f"{label} waiting for data from core ", core , " / ", self.numcores)
                for k,label in enumerate(self.vectors.keys()):
                    expected_shape = (self.numobj[label][core],)+self.columns[label]
                    data_vessel = np.empty(expected_shape, dtype=np.float64)
                    self.comm.Recv(data_vessel, source=core, tag=self.tag_start+k)
                    try:
                        self.vectors[label] = np.append(self.vectors[label],data_vessel,axis=0)
                    except: # in case rank 0 has no data because it is not participating
                        self.vectors[label] = data_vessel

            if not(skip_stats):
                for k,label in enumerate(self.vectors.keys()):
                    self.stats[label] = get_stats(self.vectors[label])
            #self.vectors = {}

    
    def dump(self,path):
        for d,name in zip([self.vectors,self.stacks],['vectors','stack']):
            for key in d.keys():
                np.save(f"{path}/mstats_dump_{name}_{key}.npy",d[key])
        for key in self.stats.keys():
            for skey in self.stats[key].keys():
                np.savetxt(f"{path}/mstats_dump_stats_{key}_{skey}.txt",np.atleast_1d(self.stats[key][skey]))
        
def load_stats(path):                
    import glob,re
    class S:
        pass
    s = S()
    s.vectors = {}
    s.stats = {}
    s.stacks = {}
    for sstr,sdict in zip(['vectors','stack'],[s.vectors,s.stacks]):
        vfiles = glob.glob(f"{path}/mstats_dump_{sstr}_*.npy")
        for vfile in vfiles:
            key = re.search(rf'mstats_dump_{sstr}_(.*?).npy', vfile).group(1)
            sdict[key] = np.load(f"{path}/mstats_dump_{sstr}_{key}.npy")

    vfiles = glob.glob(f"{path}/mstats_dump_stats_*_mean.txt")
    keys = []
    for vfile in vfiles:
        key = re.search(rf'mstats_dump_stats_(.*?)_mean.txt', vfile).group(1)
        keys.append(key)
    for key in keys:
        s.stats[key] = {}
        vfiles = glob.glob(f"{path}/mstats_dump_stats_{key}_*.txt")
        for vfile in vfiles:
            skey = re.search(rf'mstats_dump_stats_{key}_(.*?).txt', vfile).group(1)
            arr = np.loadtxt(f"{path}/mstats_dump_stats_{key}_{skey}.txt")
            if arr.size==1: arr = arr.ravel()[0]
            s.stats[key][skey] = arr
    return s

    
def npspace(minim,maxim,num,scale="lin"):
    if scale=="lin" or scale=="linear":
        return np.linspace(minim,maxim,num)
    elif scale=="log":
        return np.logspace(np.log10(minim),np.log10(maxim),num)


class bin2D(object):
    def __init__(self, modrmap, bin_edges):
        self.centers = (bin_edges[1:]+bin_edges[:-1])/2.
        self.digitized = np.digitize(modrmap.reshape(-1), bin_edges,right=True)
        self.bin_edges = bin_edges
        self.modrmap = modrmap

    def bin(self,data2d,weights=None,err=False,get_count=False,mask_nan=False):
        if weights is None:
            if mask_nan:
                keep = ~np.isnan(data2d.reshape(-1))
            else:
                keep = np.ones((data2d.size,),dtype=bool)
            count = np.bincount(self.digitized[keep])[1:-1]
            res = np.bincount(self.digitized[keep],(data2d).reshape(-1)[keep])[1:-1]/count
            if err:
                meanmap = self.modrmap.copy().reshape(-1) * 0
                for i in range(self.centers.size): meanmap[self.digitized==i] = res[i]
                std = np.sqrt(np.bincount(self.digitized[keep],((data2d-meanmap.reshape(self.modrmap.shape))**2.).reshape(-1)[keep])[1:-1]/(count-1)/count)
        else:
            count = np.bincount(self.digitized,weights.reshape(-1))[1:-1]
            res = np.bincount(self.digitized,(data2d*weights).reshape(-1))[1:-1]/count
        if get_count:
            assert not(err) # need to make more general
            return self.centers,res,count
        if err:
            assert not(get_count)
            return self.centers,res,std
        return self.centers,res


class bin1D:
    '''
    * Takes data defined on x0 and produces values binned on x.
    * Assumes x0 is linearly spaced and continuous in a domain?
    * Assumes x is continuous in a subdomain of x0.
    * Should handle NaNs correctly.
    '''
    

    def __init__(self, bin_edges):

        self.update_bin_edges(bin_edges)


    def update_bin_edges(self,bin_edges):
        
        self.bin_edges = bin_edges
        self.numbins = len(bin_edges)-1
        self.cents = (self.bin_edges[:-1]+self.bin_edges[1:])/2.

        self.bin_edges_min = self.bin_edges.min()
        self.bin_edges_max = self.bin_edges.max()

    def bin(self,ix,iy,stat=np.nanmean):
        x = ix.copy()
        y = iy.copy()
        # this just prevents an annoying warning (which is otherwise informative) everytime
        # all the values outside the bin_edges are nans
        y[x<self.bin_edges_min] = 0
        y[x>self.bin_edges_max] = 0

        # pretty sure this treats nans in y correctly, but should double-check!
        bin_means = binnedstat(x,y,bins=self.bin_edges,statistic=stat)[0]
        
        return self.cents,bin_means

        
    
def bin_in_annuli(data2d, modrmap, bin_edges):
    binner = bin2D(modrmap, bin_edges)
    return binner.bin(data2d)



def get_stats(binned_vectors):
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
    
    arr = np.asarray(binned_vectors)
    
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
        #ret['corr'] = cov2corr(ret['cov'])
    

        
    return ret



def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%r %2.2f sec' % \
              (method.__name__,te-ts))
        return result

    return timed
