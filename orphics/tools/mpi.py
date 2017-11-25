from __future__ import print_function
print("WARNING: This module is deprecated. Most of its contents have moved to orphics.mpi. If you do not find the function you require there, please raise an issue.")
import numpy as np
import os

try:
    disable_mpi_env = os.environ['DISABLE_MPI']
    disable_mpi = True if disable_mpi_env.lower().strip() == "true" else False
except:
    disable_mpi = False

try:
    if disable_mpi: raise
    from mpi4py import MPI
except:
    """
    A Simple Fake MPI implementation
    """
    class fakeMpiComm:
        def __init__(self):
            pass
        def Get_rank(self):
            return 0
        def Get_size(self):
            return 1
        def Barrier(self):
            pass

    class template:
        pass

    MPI = template()
    MPI.COMM_WORLD = fakeMpiComm()


class MPIStats(object):
    """
    A helper container for
    1) 1d measurements whose statistics need to be calculated
    2) 2d cumulative stacks

    where different MPI cores may be calculating different number
    of 1d measurements or 2d stacks.
    """
    
    def __init__(self,comm,num_each,root=0,loopover=None,tag_start=333):
        """
        comm - MPI.COMM_WORLD object
        num_each - 1d array or list where the ith element indicates number of tasks assigned to ith core
        tag_start - MPI comm tags start at this integer
        """
        
        self.comm = comm
        self.num_each = num_each
        self.rank = comm.Get_rank()
        self.numcores = comm.Get_size()    
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
                
def mpi_distribute(num_tasks,avail_cores):
    import numpy as np

    assert avail_cores<=num_tasks
    min_each, rem = divmod(num_tasks,avail_cores)
    num_each = np.array([min_each]*avail_cores) # first distribute equally
    if rem>0: num_each[-rem:] += 1  # add the remainder to the last set of cores (so that rank 0 never gets extra jobs)

    task_range = list(range(num_tasks)) # the full range of tasks
    cumul = np.cumsum(num_each).tolist() # the end indices for each task
    task_dist = [task_range[x:y] for x,y in zip([0]+cumul[:-1],cumul)] # a list containing the tasks for each core
    return num_each,task_dist
    
