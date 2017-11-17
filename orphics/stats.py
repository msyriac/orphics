import numpy as np


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
        num_each - 1d array or list where the ith element indicates number of tasks assigned to ith core
        tag_start - MPI comm tags start at this integer
        """

        if comm is not None:
            self.comm = comm
        else:
            from orphics.mpi import fakeMpiComm
            self.comm = fakeMpiComm()

            
        self.rank = self.comm.Get_rank()
        self.numcores = self.comm.Get_size()
            
            
        self.vectors = {}
        self.little_stack = {}
        self.little_stack_count = {}
        self.tag_start = tag_start
        self.root = root
        if loopover is None:
            self.loopover = list(range(root+1,self.numcores))
        else:
            self.loopover = loopover

    def add_to_stats(self,label,vector):
        """
        Append the 1d vector to a statistic named "label".
        Create a new one if it doesn't already exist.
        """
        
        if not(label in list(self.vectors.keys())): self.vectors[label] = []
        self.vectors[label].append(vector)


    def add_to_stack(self,label,arr):
        """
        This is just an accumulator, it can't track statisitics.
        Add arr to a cumulative stack named "label". Could be 2d arrays.
        Create a new one if it doesn't already exist.
        """
        if not(label in list(self.little_stack.keys())):
            self.little_stack[label] = 0.
            self.little_stack_count[label] = 0
        self.little_stack[label] += arr
        self.little_stack_count[label] += 1


    def get_stacks(self,verbose=True):
        """
        Collect from all MPI cores and calculate stacks.
        """
        import numpy as np

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
                
    def get_stats(self,verbose=True):
        """
        Collect from all MPI cores and calculate statistics for
        1d measurements.
        """
        import orphics.tools.stats as stats
        import numpy as np

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
                    expected_shape = (self.numobj[label][core],self.vectors[label].shape[1])
                    data_vessel = np.empty(expected_shape, dtype=np.float64)
                    self.comm.Recv(data_vessel, source=core, tag=self.tag_start+k)
                    self.vectors[label] = np.append(self.vectors[label],data_vessel,axis=0)

            for k,label in enumerate(self.vectors.keys()):
                self.stats[label] = stats.getStats(self.vectors[label])
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
    def bin(self,data2d):
        data = np.ndarray.flatten(data2d)
        return self.centers,np.array([np.nanmean(data[self.digitized == i]) for i in range(1, len(self.bin_edges))])

    
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
