from __future__ import print_function
import numpy as np
import time
import itertools
import scipy
from scipy.stats import binned_statistic as binnedstat

try:
    from pandas import DataFrame
    import pandas as pd
except:
    import warnings
    warnings.warn("Could not find pandas. Install using pip; FisherMatrix will not work otherwise")
    class DataFrame:
        pass

class Solver(object):
    def __init__(self,C,u=None):
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

    
def check_fisher_sanity(fmat,param_list):
    Ny,Nx = fmat.shape
    assert Ny==Nx
    assert Ny==len(param_list)
    assert len(param_list)==len(set(param_list))


class FisherMatrix(DataFrame):
    """
    A Fisher Matrix object that subclasses pandas.DataFrame.
    This is essentially just a structured array that
    has identical column and row labels.

    You can initialize an empty one like:
    >> params = ['H0','om','sigma8']
    >> F = FisherMatrix(np.zeros((len(params),len(params))),params)
    
    where params is a list of parameter names. If you already have a
    Fisher matrix 'Fmatrix' whose diagonal parameter order is specified by
    the list 'params', then you can initialize this object as:
    
    >> F = FisherMatrix(Fmatrix,params)
    
    This makes the code 'aware' of the parameter order in a way that makes
    handling combinations of Fishers a lot easier.
    
    You can set individual elements like:
    
    >> F['s8']['H0'] = 1.

    Once you've populated the entries, you can do things like:
    >> Ftot = F1 + F2
    i.e. add Fisher matrices. The nice property here is that you needn't
    have initialized these objects with the same list of parameters!
    They can, for example, have mutually exclusive parameters, in which
    case you end up with some reordering of a block diagonal Fisher matrix.
    In the general case, of two overlapping parameter lists that don't
    have the same ordering, pandas will make sure the objects are added
    correctly.

    WARNING: No other operation other than addition is overloaded. Subtraction
    for instance will give unpredictable behaviour. (Will likely introduce
    NaNs) But you shouldn't be subtracting Fisher matrices anyway!

    You can add a gaussian prior to a parameter:
    >> F.add_prior('H0',2.0)

    You can drop an entire parameter (which removes that row and column):
    >> F.delete('s8')
    which does it in place.

    If you want to preserve the original before modifying, you can
    >> Forig = F.copy()

    You can get marginalized errors on each parameter as a dict:
    >> sigmas = F.sigmas()


    """

    
    def __init__(self,fmat,param_list,delete_params=None,prior_dict=None,skip_inv=False):
        """
        fmat            -- (n,n) shape numpy array containing initial Fisher matrix for n parameters
        param_list      -- n-element list specifying diagonal order of fmat
        delete_params   -- list of names of parameters you would like to delete from this 
                        Fisher matrix when you initialize it. This is useful when skip_inv=False if some
                        of your parameters are not constrained. See skip_inv below.
        prior_dict      -- a dictionary that maps names of parameters to 1-sigma prior values
                        you would like to add on initialization. This can also be done later with the 
                        add_prior function.
        skip_inv        -- If true, this skips calculation of the inverse of the Fisher matrix
                        when the object is initialized.
	"""
	
	
        check_fisher_sanity(fmat,param_list)
        pd.DataFrame.__init__(self,fmat.copy(),columns=param_list,index=param_list)
        self.params = param_list
            
        cols = self.columns.tolist()
        ind = self.index.tolist()
        assert set(self.params)==set(cols)
        assert set(self.params)==set(ind)

        if delete_params is not None:
            self.delete(delete_params)
        if prior_dict is not None:
            for prior in prior_dict.keys():
                self.add_prior(prior,prior_dict[prior])

        self._changed = True
        if not(skip_inv):
            self._update()

            
    def copy(self, order='K'):
	"""
	>> Fnew = F.copy()
	will create an independent Fnew that is not a view of the original.
	"""
        self._update()
	f = FisherMatrix(pd.DataFrame.copy(self), list(self.params),skip_inv=True)
        f._finv = self._finv
        f._changed = False
        return f

    def _update(self):
        if self._changed:
            self._finv = np.linalg.inv(self.as_matrix())
            self._changed = False
        
    def __radd__(self,other):
        return self._add(other,radd=True)

    def __add__(self,other):
        return self._add(other,radd=False)

    def _add(self,other,radd=False):
        if radd:
            new_fpd = pd.DataFrame.radd(self,other.copy(),fill_value=0)
        else:
            new_fpd = pd.DataFrame.add(self,other.copy(),fill_value=0)
        return FisherMatrix(np.nan_to_num(new_fpd.as_matrix()),new_fpd.columns.tolist())

    def add_prior(self,param,prior):
	"""
	Adds 1-sigma value 'prior' to the parameter name specified by 'param'
	"""
        self[param][param] += 1./prior**2.
        self._changed = True
        
    def sigmas(self):
	"""
	Returns marginalized 1-sigma uncertainties on each parameter in the Fisher matrix.
	"""
        self._update()
        errs = np.diagonal(self._finv)**(0.5)
        return dict(zip(self.params,errs))
    
    def delete(self,params):
	"""
	Given a list of parameter names 'params', deletes these from the Fisher matrix.
	"""
        self.drop(labels=params,axis=0,inplace=True)
        self.drop(labels=params,axis=1,inplace=True)
        self.params = self.columns.tolist()
        assert set(self.index.tolist())==set(self.params)
        self._changed = True

    def marge_var_2param(self,param1,param2):
	"""
	Returns the sub-matrix corresponding to two parameters param1 and param2.
	Useful for contour plots.
	"""
        self._update()
        i = self.params.index(param1)
        j = self.params.index(param2)
        chi211 = self._finv[i,i]
        chi222 = self._finv[j,j]
        chi212 = self._finv[i,j]
        
        return np.array([[chi211,chi212],[chi212,chi222]])
        

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
        #return scipy.linalg.pinv2(cov)


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

def cov2corr(cov):
    # slow and stupid!
    
    d = np.diag(cov)
    stddev = np.sqrt(d)
    corr = cov.copy()*0.
    for i in range(cov.shape[0]):
        for j in range(cov.shape[0]):
            corr[i,j] = cov[i,j]/stddev[i]/stddev[j]

    return corr

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
                for core in self.loopover: #range(1,self.numcores):
                    if verbose: print("Waiting for core ", core , " / ", self.numcores)
                    data = self.comm.recv(source=core, tag=self.tag_start*3000+k)
                    self.stack_count[label] += data

            
            for k,label in enumerate(self.little_stack.keys()):
                self.stacks[label] = self.little_stack[label]
            for core in self.loopover: #range(1,self.numcores):
                if verbose: print("Waiting for core ", core , " / ", self.numcores)
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
                    if verbose: print("Waiting for core ", core , " / ", self.numcores)
                    data = self.comm.recv(source=core, tag=self.tag_start*2000+k)
                    self.numobj[label].append(data)

            
            for k,label in enumerate(self.vectors.keys()):
                self.vectors[label] = np.array(self.vectors[label])
            for core in self.loopover: #range(1,self.numcores):
                if verbose: print("Waiting for core ", core , " / ", self.numcores)
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
                


def npspace(minim,maxim,num,scale="lin"):
    if scale=="lin" or scale=="linear":
        return np.linspace(minim,maxim,num)
    elif scale=="log":
        return np.logspace(np.log10(minim),np.log10(maxim),num)


class bin2D(object):
    def __init__(self, modrmap, bin_edges):
        self.centers = (bin_edges[1:]+bin_edges[:-1])/2.
        self.digitized = np.digitize(np.ndarray.flatten(modrmap), bin_edges,right=True)
        self.bin_edges = bin_edges
    def bin(self,data2d,weights=None):
        
        if weights is None:
            res = np.bincount(self.digitized,(data2d).reshape(-1))[1:-1]/np.bincount(self.digitized)[1:-1]
        else:
            #weights = self.digitized*0.+weights
            res = np.bincount(self.digitized,(data2d*weights).reshape(-1))[1:-1]/np.bincount(self.digitized,weights.reshape(-1))[1:-1]
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


    def binned(self,x,y):


        # pretty sure this treats nans in y correctly, but should double-check!
        bin_means = binnedstat(x,y,bins=self.bin_edges,statistic=np.nanmean)[0]


        
        return (self.bin_edges[:-1]+self.bin_edges[1:])/2.,bin_means

        
    
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
