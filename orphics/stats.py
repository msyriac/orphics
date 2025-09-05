from __future__ import print_function
import numpy as np
import time, warnings
import itertools
import scipy
from scipy.stats import binned_statistic as binnedstat,chi2
from scipy.optimize import curve_fit
import itertools
from collections import defaultdict
from typing import Any, Dict, Hashable, Optional, Tuple
from pathlib import Path
try:
    from mpi4py import MPI
except Exception:
    MPI = None


def extrapolate_power_law(x, y, x_extra, x_percentile=30.0):
    power_law = lambda x, a, b:  a * x**b

    # Only use large x values for fitting (e.g., top 30%)
    threshold = np.percentile(x, 100-x_percentile)
    mask = x >= threshold
    x_tail = x[mask]
    y_tail = y[mask]

    # Fit the power law
    params, _ = curve_fit(power_law, x_tail, y_tail)
    a, b = params

    # Extrapolate to higher x
    y_extra = power_law(x_extra, a, b) 

    xout = np.append(x, x_extra)
    yout = np.append(y, y_extra)

    return xout, yout

def nsigma_from_pte(pte):
    from scipy.special import erfinv
    return erfinv ( (1-pte)) * np.sqrt(2)

def get_pte(chisquare_data,chisquares_sims):
    return chisquares_sims[chisquare_data<chisquares_sims].size / chisquares_sims.size

# Get PTE from samples drawn from a covariance matrix
def sim_pte(data,covmat,nsamples):
    cinv = np.linalg.inv(covmat)
    chisquare = np.dot(data,np.dot(cinv,data))
    samples = np.random.multivariate_normal(data*0,covmat,nsamples)
    chisquares = np.einsum('ik,ik->i', np.einsum('ij,jk->ik',samples,cinv),samples)
    pte = get_pte(chisquare,chisquares)
    return pte

class InverseTransformSampling(object):
    # Sample from an arbitrary 1d PDF
    def __init__(self,xvals,pdf_vals):
        from scipy.interpolate import interp1d
        
        # spacing between x-values
        dxs = np.diff(xvals)
        if not( np.all(np.isclose(dxs,dxs[0])) ):
            # It might not have to be, but I haven't tested beyond equi-spaced.
            raise Exception("The PDF domain has to be equi-spaced.")
        
        # Normalize
        norm = np.trapz(pdf_vals,xvals)
        self.xs = xvals
        self.pdf = pdf_vals/norm

        # CDF
        self.cdf = np.cumsum(self.pdf)*dxs[0]
        # Make it sensible. The discreteness will make the behaviour in
        # the tails not quite right.
        self.cdf[0] = 0
        self.cdf[self.cdf>1] = 1
        if not(np.all(self.cdf>=0)): raise Exception
        if not(np.all(self.cdf<=1)): raise Exception
        # inverse CDF
        self.icdf = interp1d(self.cdf,self.xs,bounds_error=False)
        
    def generate(self,nsamples):
        # CDF^{-1} ( U(0,1) )
        return self.icdf(np.random.uniform(0,1,size=nsamples))



class InverseTransformSampling2D(object):
    # Sample from an arbitrary 2d PDF
    def __init__(self,ys,xs,updf,bounds_error=False):
        self.ys = ys
        
        # Normalize PDF p(y,x)
        norm = np.trapz(np.trapz(updf,xs),ys)
        pdf = updf / norm
        self.pdf = pdf

        # Marginal PDF p(y)
        mpdf_y = np.trapz(pdf,xs)
        # Prepare to sample from p(y)
        self.its = InverseTransformSampling(ys,mpdf_y)

        # Conditional density p(x | y)
        cpdf = (pdf.T / mpdf_y).T

        # Prepare to sample from p(x | y) for all y
        self.allits = []
        for i in range(len(ys)):
            self.allits.append(InverseTransformSampling(xs,cpdf[i,:]) )

    def generate(self,nsamples):
        # Sample from p(y)
        ysamples = np.asarray(self.its.generate(nsamples))
        # Find index in ys corresponding to sampled y
        # and sample x from corresponding p(x | y)
        diff = np.abs(self.ys-ysamples[:,None])
        inds = np.argmin(diff,axis=1)
        xsamples = np.asarray([self.allits[ind].generate(1)[0] for ind in inds])
        return ysamples,xsamples


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
        

def fit_linear_model(x,y,ycov,funcs,dofs=None,deproject=False,Cinv=None,Cy=None):
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
    dofs = len(x)-len(funcs) if dofs is None else dofs
    pte = 1 - chi2.cdf(chisquare, dofs)    
    return X,cov,chisquare/dofs,pte

def fit_linear_model_pte_from_sims(x,y,ycov,funcs,y_fiducial,nsims=100000,deproject=False,Cinv=None,Cy=None):
    X_data,cov_data,chisquare_data,_ = fit_linear_model(x,y,ycov,funcs,dofs=None,deproject=False)
    samples = y_fiducial + np.random.multivariate_normal(y_fiducial*0,ycov,nsims)
    chisquares_sims = []
    for i in range(nsims):
        _,_,chisq,_ = fit_linear_model(x,samples[i],ycov,funcs,dofs=None,deproject=deproject,Cinv=Cinv,Cy=Cy)
        chisquares_sims.append(chisq)
    chisquares_sims = np.asarray(chisquares_sims)
    pte = get_pte(chisquare_data,chisquares_sims)
    return X_data,cov_data,chisquare_data,pte 

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

    fishers -- list of pyfisher.FisherMatrix objects
    labels -- labels corresponding to each Fisher matrix in fishers
    params -- By default (if None) uses all params in every Fisher matrix. If not None, uses only this list of params.
    fid_dict -- dictionary mapping parameter names to fiducial values to center ellipses on
    latex_dict -- dictionary mapping parameter names to LaTeX strings for axes labels. Defaults to parameter names.
    confidence_level -- fraction of probability density enclosed by ellipse
    colors -- list of colors corresponding to fishers
    lss -- list of line styles corresponding to fishers
    """

    from . import io
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
                if verbose: print(f"Waiting for data from core ", core , " / ", self.numcores)
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
                if verbose: print(f"Waiting for data from core ", core , " / ", self.numcores)
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
        self.cents = self.centers # backwards compatibility
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
        ret['corr'] = cov2corr(ret['cov'])
    

        
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




class Statistics:
    """
    Improved version of Stats: MPI-aware container for accumulating statistics or elementwise sums of arrays.

    This class supports two mutually-exclusive accumulation modes, per label:

    1. **Stats mode** (via :meth:`add` or :meth:`extend`):
       - Accepts 1-D sample vectors (shape ``(d,)``).
       - Tracks the sample count, sum vector, and second-moment cross product.
       - After MPI allreduce, allows queries for mean and covariance.

    2. **Stack mode** (via :meth:`add_stack`):
       - Accepts arbitrary-shape arrays.
       - Tracks only the elementwise sum and number of stacked arrays.
       - After MPI allreduce, allows queries for the stack sum and count.

    Each label (a hashable identifier) can be assigned to exactly one mode
    (stats or stack). The chosen mode must be consistent across all MPI ranks.

   Example (stats mode with MPI, analytic check)::

        from mpi4py import MPI
        import numpy as np

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Each rank contributes the vector [rank+1, 2*(rank+1)]
        x = np.array([rank + 1, 2 * (rank + 1)], dtype=float)

        stats = Statistics(comm)
        stats.add("vectors", x)
        stats.allreduce()

        # Analytic expectations
        # Global count = number of ranks
        expected_count = size

        # Sum over ranks = sum_{r=1..size} [r, 2r]
        expected_sum = np.array([
            size * (size + 1) / 2,
            2 * size * (size + 1) / 2
        ])
        expected_mean = expected_sum / expected_count

        assert stats.count("vectors") == expected_count
        assert np.allclose(stats.mean("vectors"), expected_mean)
    
    Example (stack mode with MPI, analytic check)::

        from mpi4py import MPI
        import numpy as np

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        base = np.ones((2,2))
        arr = (rank + 1) * base

        stats = Statistics(comm)
        stats.add_stack("matrices", arr)
        stats.allreduce()

        expected_sum = base * (size * (size+1) / 2)
        assert np.allclose(stats.stack_sum("matrices"), expected_sum)
        assert stats.stack_count("matrices") == size


    Parameters
    ----------
    comm : mpi4py.MPI.Comm, optional
        The MPI communicator. If None, runs in single-process mode.
    dtype : numpy dtype, default numpy.float64
        Data type for all accumulated values.
    
    """

    def __init__(self, comm: Optional[Any] = None, dtype=np.float64):
        """
        Initialize an empty Statistics container.

        Parameters
        ----------
        comm : mpi4py.MPI.Comm, optional
            The MPI communicator. If None, runs in single-process mode.
        dtype : numpy dtype, default numpy.float64
            Data type for all accumulated values.
        """
        self.comm = comm
        self.dtype = np.dtype(dtype)

        # Stats-mode partials
        self._n: Dict[Hashable, int] = defaultdict(int)
        self._sum: Dict[Hashable, np.ndarray] = {}
        self._cross: Dict[Hashable, np.ndarray] = {}
        self._dim_stats: Dict[Hashable, int] = {}

        # Stack-mode partials
        self._k: Dict[Hashable, int] = defaultdict(int)                 # local count of stacked objects
        self._stack_sum: Dict[Hashable, np.ndarray] = {}                # local sum over stacked arrays
        self._shape_stack: Dict[Hashable, Tuple[int, ...]] = {}         # declared shape per label

        # Global reduced results
        self._N: Dict[Hashable, int] = {}
        self._SUM: Dict[Hashable, np.ndarray] = {}
        self._CROSS: Dict[Hashable, np.ndarray] = {}

        self._K: Dict[Hashable, int] = {}
        self._STACK_SUM: Dict[Hashable, np.ndarray] = {}

        self._reduced = False

    # utilities
    @property
    def mpi_enabled(self) -> bool:
        """
        Whether MPI is enabled for this instance.

        Returns
        -------
        bool
            True if an MPI communicator is provided and mpi4py is available,
            otherwise False.
        """
        return (self.comm is not None) and (MPI is not None)

    def _ensure_stats_label(self, label: Hashable, d: int):
        # prevent mixing modes on same label
        if label in self._shape_stack:
            raise ValueError(f"Label {label!r} already used in stack mode.")
        if label not in self._dim_stats:
            self._dim_stats[label] = int(d)
            self._sum[label] = np.zeros(d, dtype=self.dtype)
            self._cross[label] = np.zeros((d, d), dtype=self.dtype)
        elif self._dim_stats[label] != d:
            raise ValueError(f"Stats dim mismatch for {label!r}: {self._dim_stats[label]} vs {d}")

    def _ensure_stack_label(self, label: Hashable, shape: Tuple[int, ...]):
        # prevent mixing modes on same label
        if label in self._dim_stats:
            raise ValueError(f"Label {label!r} already used in stats mode.")
        if label not in self._shape_stack:
            self._shape_stack[label] = tuple(int(s) for s in shape)
            self._stack_sum[label] = np.zeros(shape, dtype=self.dtype)
        elif self._shape_stack[label] != tuple(shape):
            raise ValueError(f"Stack shape mismatch for {label!r}: {self._shape_stack[label]} vs {tuple(shape)}")

    # ingestion: stats mode
    def add(self, label: Hashable, x: Any):
        """
        Add a single 1-D sample vector to a stats-mode label.

        Parameters
        ----------
        label : hashable
            The label under which to accumulate statistics.
        x : array_like, shape (d,)
            Sample vector to add.

        Raises
        ------
        ValueError
            If `x` is not 1-D or if label is already in stack mode.
        """
        x = np.asarray(x, dtype=self.dtype).ravel()
        if x.ndim != 1:
            raise ValueError("Samples must be 1-D vectors.")
        d = x.shape[0]
        self._ensure_stats_label(label, d)
        self._n[label] += 1
        self._sum[label] += x
        self._cross[label] += np.outer(x, x)

    def extend(self, label: Hashable, X: Any):
        """
        Add multiple samples to a stats-mode label.

        Parameters
        ----------
        label : hashable
            The label under which to accumulate statistics.
        X : array_like, shape (m, d) or (d,)
            - If shape (m, d): m samples of dimension d.
            - If shape (d,): equivalent to calling :meth:`add`.

        Raises
        ------
        ValueError
            If `X` is not 1-D or 2-D, or if label is already in stack mode.
        """
        X = np.asarray(list(X) if not hasattr(X, "shape") else X, dtype=self.dtype)
        if X.ndim == 1:
            self.add(label, X)
            return
        if X.ndim != 2:
            raise ValueError("X must be (m, d) or (d,).")
        m, d = X.shape
        self._ensure_stats_label(label, d)
        self._n[label] += m
        self._sum[label] += X.sum(axis=0)
        self._cross[label] += X.T @ X   # BLAS-backed Gram update

    # ingestion: stack mode
    def add_stack(self, label: Hashable, arr: Any):
        """
        Add an arbitrary-shape array to a stack-mode label.

        Only the elementwise sum and count of arrays are tracked.

        Parameters
        ----------
        label : hashable
            The label under which to accumulate the stack sum.
        arr : array_like
            Array to add. May have any shape; must be consistent per label.

        Raises
        ------
        ValueError
            If array shape mismatches previous arrays for the same label,
            or if label is already in stats mode.
        """
        A = np.asarray(arr, dtype=self.dtype)
        if A.ndim == 0:
            # Allow scalars, treat as shape ()
            shape = ()
        else:
            shape = A.shape
        self._ensure_stack_label(label, shape)
        self._k[label] += 1
        self._stack_sum[label] += A

    # reduction helpers
    def _union_dims(self):
        """
        Build union of label->dim/shape for both modes across ranks.
        Ensures all ranks have consistent dims/shapes for each label.
        """
        local = {
            "stats": [(lab, d) for lab, d in self._dim_stats.items()],
            "stack": [(lab, shp) for lab, shp in self._shape_stack.items()],
        }
        if self.mpi_enabled:
            all_lists = self.comm.allgather(local)
            stats_union: Dict[Hashable, int] = {}
            stack_union: Dict[Hashable, Tuple[int, ...]] = {}
            for entry in all_lists:
                for lab, d in entry["stats"]:
                    if lab in stats_union and stats_union[lab] != d:
                        raise ValueError(f"Stats dim mismatch for {lab!r} across ranks.")
                    if (lab in stack_union):
                        raise ValueError(f"Label {lab!r} used in stats and stack across ranks.")
                    stats_union[lab] = d
                for lab, shp in entry["stack"]:
                    shp = tuple(shp)
                    if lab in stack_union and stack_union[lab] != shp:
                        raise ValueError(f"Stack shape mismatch for {lab!r} across ranks.")
                    if (lab in stats_union):
                        raise ValueError(f"Label {lab!r} used in stats and stack across ranks.")
                    stack_union[lab] = shp
            return stats_union, stack_union
        else:
            return dict(self._dim_stats), dict(self._shape_stack)

    def allreduce(self):
        """
        Perform an MPI allreduce to combine contributions across all ranks.

        After calling this, the global statistics or stack sums can be queried.

        Reductions performed:
        - Stats mode: sample counts, sums, and cross products.
        - Stack mode: stack counts and elementwise sums.

        Raises
        ------
        ValueError
            If different ranks disagree on label dimensions/shapes or mode.
        """
        stats_union, stack_union = self._union_dims()

        # Ensure zero entries exist locally for all globally known labels
        for lab, d in stats_union.items():
            if lab not in self._dim_stats:
                self._ensure_stats_label(lab, d)  # creates zero arrays; n remains 0
        for lab, shp in stack_union.items():
            if lab not in self._shape_stack:
                self._ensure_stack_label(lab, shp)  # creates zero arrays; k remains 0

        # Reduce stats
        for lab, d in stats_union.items():
            n_loc = np.array(self._n.get(lab, 0), dtype=np.int64)
            sum_loc = self._sum[lab]
            cross_loc = self._cross[lab]
            if self.mpi_enabled:
                self.comm.Allreduce(MPI.IN_PLACE, n_loc, op=MPI.SUM)
                self.comm.Allreduce(MPI.IN_PLACE, sum_loc, op=MPI.SUM)
                self.comm.Allreduce(MPI.IN_PLACE, cross_loc, op=MPI.SUM)
            self._N[lab] = int(n_loc)
            self._SUM[lab] = sum_loc
            self._CROSS[lab] = cross_loc

        # Reduce stack
        for lab, shp in stack_union.items():
            k_loc = np.array(self._k.get(lab, 0), dtype=np.int64)
            stk_loc = self._stack_sum[lab]
            if self.mpi_enabled:
                self.comm.Allreduce(MPI.IN_PLACE, k_loc, op=MPI.SUM)
                self.comm.Allreduce(MPI.IN_PLACE, stk_loc, op=MPI.SUM)
            self._K[lab] = int(k_loc)
            self._STACK_SUM[lab] = stk_loc

        self._reduced = True

    # queries
    def labels_stats(self):
        """
        Return labels currently in stats mode.

        Returns
        -------
        list of hashable
            Labels in stats mode. After :meth:`allreduce`, includes all labels
            seen across all ranks.
        """
        return list(self._SUM.keys()) if self._reduced else list(self._dim_stats.keys())

    def labels_stack(self):
        """
        Return labels currently in stack mode.

        Returns
        -------
        list of hashable
            Labels in stack mode. After :meth:`allreduce`, includes all labels
            seen across all ranks.
        """
        return list(self._STACK_SUM.keys()) if self._reduced else list(self._shape_stack.keys())

    def count(self, label: Hashable) -> int:
        """
        Return the global sample count for a stats-mode label.

        Parameters
        ----------
        label : hashable
            The label of interest.

        Returns
        -------
        int
            Number of samples across all ranks.

        Raises
        ------
        KeyError
            If the label is not in stats mode.
        RuntimeError
            If called before :meth:`allreduce`.
        """
        self._check_reduced()
        if label not in self._N:
            raise KeyError(f"{label!r} is not a stats-mode label.")
        return self._N[label]

    def stack_count(self, label: Hashable) -> int:
        """
        Return the global count of arrays stacked under a label.

        Parameters
        ----------
        label : hashable
            The label of interest.

        Returns
        -------
        int
            Number of arrays stacked across all ranks.

        Raises
        ------
        KeyError
            If the label is not in stack mode.
        RuntimeError
            If called before :meth:`allreduce`.
        """
        self._check_reduced()
        if label not in self._K:
            raise KeyError(f"{label!r} is not a stack-mode label.")
        return self._K[label]

    def mean(self, label: Hashable) -> np.ndarray:
        """
        Compute the global mean vector for a stats-mode label.

        Parameters
        ----------
        label : hashable
            The label of interest.

        Returns
        -------
        ndarray, shape (d,)
            Mean vector. Returns NaN if count is zero.

        Raises
        ------
        KeyError
            If the label is not in stats mode.
        RuntimeError
            If called before :meth:`allreduce`.
        """
        self._check_reduced()
        if label not in self._SUM:
            raise KeyError(f"{label!r} is not a stats-mode label.")
        n = self._N[label]
        return self._SUM[label] / n if n > 0 else np.full(self._SUM[label].shape, np.nan, dtype=self.dtype)

    def cov(self, label: Hashable, ddof: int = 1) -> np.ndarray:
        """
        Compute the global covariance matrix for a stats-mode label.

        Parameters
        ----------
        label : hashable
            The label of interest.
        ddof : int, default=1
            Delta degrees of freedom. The divisor is (n - ddof).

        Returns
        -------
        ndarray, shape (d, d)
            Covariance matrix. Returns NaN if insufficient samples.

        Raises
        ------
        KeyError
            If the label is not in stats mode.
        RuntimeError
            If called before :meth:`allreduce`.
        """
        self._check_reduced()
        if label not in self._CROSS:
            raise KeyError(f"{label!r} is not a stats-mode label.")
        n = self._N[label]
        if n <= ddof:
            d = self._SUM[label].shape[0]
            return np.full((d, d), np.nan, dtype=self.dtype)
        S = self._SUM[label]
        C = self._CROSS[label]
        return (C - np.outer(S, S) / n) / (n - ddof)

    def var(self, label, ddof: int = 1) -> np.ndarray:
        """
        Per-dimension variance for a stats-mode label.
        ddof=1 -> sample variance; ddof=0 -> population variance.

        Requires .allreduce() to have been called.
        """
        self._check_reduced()
        if label not in self._CROSS:
            raise KeyError(f"{label!r} is not a stats-mode label.")
        n = self._N[label]
        if n <= ddof:
            d = self._SUM[label].shape[0]
            return np.full(d, np.nan, dtype=self.dtype)

        S = self._SUM[label]                   # shape (d,)
        C = self._CROSS[label]                 # shape (d,d)
        # diagonal of centered second moment: diag(C - (1/n) * S S^T)
        # = diag(C) - (S**2)/n
        centered_diag = np.diag(C) - (S * S) / n
        return centered_diag / (n - ddof)
    

    def stack_sum(self, label: Hashable) -> np.ndarray:
        """
        Return the global elementwise sum of arrays stacked under a label.

        Parameters
        ----------
        label : hashable
            The label of interest.

        Returns
        -------
        ndarray
            Array of same shape as inputs to :meth:`add_stack`.

        Raises
        ------
        KeyError
            If the label is not in stack mode.
        RuntimeError
            If called before :meth:`allreduce`.
        """
        self._check_reduced()
        if label not in self._STACK_SUM:
            raise KeyError(f"{label!r} is not a stack-mode label.")
        return self._STACK_SUM[label]

    def _check_reduced(self):
        if not self._reduced:
            raise RuntimeError("Call .allreduce() before requesting global stats/stack.")

    def _to_state(self) -> dict:
        """
        Build a versioned, pickle-friendly snapshot of the *reduced* stats.
        Requires that .allreduce() has already been called.
        """
        self._check_reduced()
        # Store dtype as NumPy dtype string (e.g., '<f8')
        dtype_str = str(self.dtype.str)

        # Convert keys to a list to preserve a deterministic order on reload
        stats_labels = list(self._SUM.keys())
        stack_labels = list(self._STACK_SUM.keys())

        state = {
            "version": 1,
            "dtype": dtype_str,
            "stats": {
                "labels": stats_labels,
                "N": {lab: int(self._N[lab]) for lab in stats_labels},
                "SUM": {lab: self._SUM[lab] for lab in stats_labels},       # (d,)
                "CROSS": {lab: self._CROSS[lab] for lab in stats_labels},   # (d,d)
            },
            "stack": {
                "labels": stack_labels,
                "SUM": {lab: self._STACK_SUM[lab] for lab in stack_labels},  # arbitrary shape
            },
        }
        return state


    def save_reduced(self, path: str | Path, compressed: bool = False, root_rank=0):
        """
        Save reduced statistics to a .npz file (portable, numpy-only), if root.

        Parameters
        ----------
        path : str or Path
            Destination file.
        compressed : bool, default True
            Use np.savez_compressed (smaller, slower) vs np.savez (larger, faster).
        """
        self._check_reduced()
        if self.mpi_enabled and not (self.comm.Get_rank()==root_rank):
            return

        arrays = {}
        # stats labels
        for lab in self._SUM.keys():
            key_base = f"stats/{lab}"
            arrays[f"{key_base}/N"] = np.array(self._N[lab], dtype=np.int64)
            arrays[f"{key_base}/SUM"] = self._SUM[lab]
            arrays[f"{key_base}/CROSS"] = self._CROSS[lab]
        # stack labels
        for lab in self._STACK_SUM.keys():
            key_base = f"stack/{lab}"
            arrays[f"{key_base}/SUM"] = self._STACK_SUM[lab]

        path = Path(path)
        saver = np.savez_compressed if compressed else np.savez
        saver(path, **arrays)

    @classmethod
    def load_reduced(cls, path: str | Path, comm=None, dtype=np.float64):
        """
        Load reduced statistics from a .npz file created by save_reduced_npz.

        Parameters
        ----------
        path : str or Path
            Source file.
        comm : MPI.Comm or None
            Optional MPI communicator for the returned object.

        Returns
        -------
        acc : LabeledVectorAllreduce
            A new accumulator with reduced stats populated.
        """
        path = Path(path)
        data = np.load(path, allow_pickle=False)

        acc = cls(comm=comm, dtype=dtype)  # dtype can be inferred if needed
        acc._N, acc._SUM, acc._CROSS = {}, {}, {}
        acc._STACK_SUM = {}
        acc._dim_stats, acc._dim_stack = {}, {}
        for key in data.files:
            parts = key.split("/")
            if parts[0] == "stats":
                lab = parts[1]
                if parts[2] == "N":
                    acc._N[lab] = int(data[key])
                elif parts[2] == "SUM":
                    arr = np.array(data[key])
                    acc._SUM[lab] = arr
                    acc._dim_stats[lab] = arr.shape[0]
                elif parts[2] == "CROSS":
                    acc._CROSS[lab] = np.array(data[key])
            elif parts[0] == "stack":
                lab = parts[1]
                if parts[2] == "SUM":
                    arr = np.array(data[key])
                    acc._STACK_SUM[lab] = arr
                    acc._dim_stack[lab] = arr.shape

        acc._reduced = True
        return acc






        
