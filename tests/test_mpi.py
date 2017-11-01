from orphics.tools.mpi import MPI
import os

# Get MPI comm
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    

print "rank ", rank , " says hello world"

if rank==0:
    print 'OMP_NUM_THREADS: ', os.environ['OMP_NUM_THREADS']
