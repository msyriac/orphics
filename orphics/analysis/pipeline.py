import numpy as np

class Pipeline(object):
    """ This class allows flexible distribution of N tasks across m MPI cores.
    'Flexible' here means that options to keep some cores idle in order to
    conserve memory are provided. This class and its subclasses will/should 
    ensure that the idle cores are not involved in MPI communications.

    In its simplest form, there are N tasks and m MPI cores. quotient(N/m) tasks 
    will be run on each core in series plus one extra job for the remainder(N/m)
    cores.

    But say you have 10 nodes with 8 cores each, and 160 tasks. You could
    say N=160 and pass the MPI Comm object which tells the class that m=80.
    But you don't want to use all cores because the node memory is limited.
    You only want to use 4 cores per node. Then you specify:

    cores_per_node = 8
    max_cores_per_node = 4

    This will "stride" the MPI cores by k = cores_per_node/max_cores_per_node = 2
    involving only ever other MPI core in communications. This means that the number
    of available cores j = m/k = 40. So the number of serial tasks is now 4 instead
    of 2.


    The base class Pipeline just implements the job distribution and striding. It is
    up to the derived classes to overload the "define_job" function and actually
    do something.
    
    """
    

    def __init__(self,MPI_Comm_World,num_tasks,cores_per_node=None,max_cores_per_node=None):

        self.mcomm = MPI_Comm_World
        self.mrank = self.mcomm.Get_rank()
        self.msize = self.mcomm.Get_size()

        maxcores = max_cores_per_node
        cpernode = cores_per_node
        msize = self.msize
        
        if maxcores is not None:
            # we want to do some striding
            assert cpernode is not None
            assert cpernode%maxcores==0

            self.stride = cpernode/maxcores # stride value
            assert msize%self.stride==0
            
        else:
            # no striding
            self.stride = 1

        self.avail_cores = msize/self.stride

        self.participants = range(0,msize,self.stride) # the cores that participate in MPI comm

        task_ids = range(num_tasks)
        min_each, rem = divmod(num_tasks,self.avail_cores)
        num_each = np.array([min_each]*self.avail_cores) # first distribute equally
        num_each[-rem:] += 1  # add the remainder to the last set of cores (so that rank 0 never gets extra jobs)

        self.num_each = np.zeros(msize,dtype=np.int)
        self.num_each[self.participants] = num_each
        self.num_each = self.num_each.tolist()
        assert sum(self.num_each)==num_tasks

        
        if self.mrank not in self.participants:
            self.idle = True
        else:
            self.idle = False

        self.num_mine = self.num_each[self.mrank]

            

    def info(self):
        if self.mrank==0:
            print "Rank 0 says the participants are ", self.participants
            print "Jobs in each core :", self.num_each
        print "==== My rank is ", self.mrank, ". Idle: ", self.idle, ". I have ", self.num_mine, " tasks. ==="
    

# pipeline = CMB_Pipeline(comm)


# class CMB_Pipeline(Pipeline):
#     def __init__(self):
#         pass

# class CMB_Lensing_Pipeline(CMB_Pipeline):
#     def __init__(self):
#         pass
    
# class CMB_Cluster_Lensing_Pipeline(CMB_Lensing_Pipeline):
#     def __init__(self):
#         pass

    
    


# def is_only_one_not_none(a):
#     """ Useful for function arguments, returns True if the list 'a'
#     contains only one non-None object and False if otherwise.

#     Examples:

#     >>> is_only_one_not_none([None,None,None])
#     >>> False
#     >>> is_only_one_not_none([None,1,1,4])
#     >>> False
#     >>> is_only_one_not_none([1,None,None,None])
#     >>> True
#     >>> is_only_one_not_none([None,None,6,None])
#     >>> True
#     """
#     return True if sum([int(x is not None) for x in a])==1 else False
    

# class CMB_Array_Patch(object):
#     """ Make one of these for each array+patch combination.
#     Maximum memory used should be = 

#     """
    

#     def __init__(self,enmap_patch_template,beam2d=None,beam_arcmin=None,beam_tuple=None,
#                  noise_uk_arcmin=None,lknee=None,alpha=None,noise2d=None,hit_map=None):

#         assert is_only_one_not_none([beam2d,beam_arcmin,beam_tuple]), "Multiple or no beam options specified"
#         if noise2d is not None:
#             # 2d noise specified
#             assert noise_uk_arcmin is None and lknee is None and alpha is None
#         else:
#             pass

# def get_unlensed_cmb(patch,cosmology,pol=False):
#     cmaps = grf.make_map(patch,cosmology,pol=pol)
#     if self.verify_unlensed_power: self.upowers.append(get_cmb_powers(cmaps,self.taper))
#     return cmaps





# if test=="liteMap_file":
#     template_liteMap_path = ""
#     lmap = liteMap.liteMapFromFits(template_liteMap_path)
#     template_enmap = enmap_from_liteMap(lmap)
    
# elif test=="wide_patch":
#     shape, wcs = enmap.geometry(pos=[[-hwidth*arcmin,-hwidth*arcmin],[hwidth*arcmin,hwidth*arcmin]], res=px*arcmin, proj="car")

    
# elif test=="cluster_patch":


# test_array = CMB_Array_Patch(template_enmap,beam_arcmin=1.4,noise_uk_arcmin=20.,lknee=3000,alpha=-4.6)
    
