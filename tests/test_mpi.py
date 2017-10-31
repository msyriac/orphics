from mpi4py import MPI

# Get MPI comm
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    

print "rank ", rank , " says hello world"

while True:
    pass
