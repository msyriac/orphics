from __future__ import print_function
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

    class template:
        pass

    MPI = template()
    MPI.COMM_WORLD = fakeMpiComm()


class fakeMpiComm:
    """
    A Simple Fake MPI implementation
    """
    def __init__(self):
        pass
    def Get_rank(self):
        return 0
    def Get_size(self):
        return 1
    def Barrier(self):
        pass


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
    
    
