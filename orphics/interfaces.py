from __future__ import print_function
import os,sys
from tempfile import mkstemp
from shutil import move,copyfile
from os import remove, close
import subprocess
import numpy as np



#TODO:
# test that different MPI jobs get different temp ini files

class CAMBInterface(object):
    """ Credit: A lot of this was written by Nam Ho Nguyen
    This interface lets you call Fortran CAMB from Python with full control over the ini parameters
    and retrieve galaxy Cls. It is primarily written with CAMB Sources in mind. TODO: Generalize

    """

    def __init__(self,ini_template,camb_loc):
        """
        ini_template is the full path to a file that will be used as the "base" ini file. Parameters can be
        modified and added relative to this ini. Usually this is the barebones "params.ini" or "params_lensing.ini"
        in the root of CAMB.

        camb_loc is the path to the directory containing the "camb" executable.

        """

        # cp ini_template to temporary
        self.ifile = ini_template.strip()[:-4]+"_itemp_"+str(os.geteuid())+".ini"
        copyfile(ini_template,self.ifile)
        self.out_name = "itemp_"+str(os.geteuid())
        self.set_param('output_root',self.out_name)

        self.camb_loc = camb_loc

    def set_param(self,param,value):
        """ Set a parameter to a certain value. If it doesn't exist, it appends to the end of the base ini file.
        """
        self._replace(self.ifile,param,subst=param+"="+str(value))
        
    def call(self,suppress=True):
        """
        Once you're done setting params, just use the call() function to run CAMB.
        Set suppress = False to get the full CAMB output.
        """
        if suppress:
            with open(os.devnull, "w") as f:
                subprocess.call([self.camb_loc+"/camb",self.ifile],stdout=f,cwd=self.camb_loc)
        else:
            subprocess.call([self.camb_loc+"/camb",self.ifile],cwd=self.camb_loc)


    def get_cls(self):
        """
        This function returns the Cls output by CAMB Sources. TODO: Generalize.
        If there are N redshift windows specified, then this returns:
        ells, clarr
        where clarr is of shape (N+3,N+3,ells.size)

        ells.size is determined by the lmax requested in the ini file
        The [i,j,:] slice is the cross-correlation of the ith component with the jth
        component in L(L+1)C/2pi form. (It is symmetric and hence redundant in i,j)
        
        The order of the components is:
        CMB T
        CMB E
        CMB phi
        redshift1
        redshift2
        ... etc.

        What the redshift components corresponds to (galaxy overdensities or galaxy lensing) 
        depends on the ini file and the set parameters. 
        """
        
        filename =self.camb_loc+"/"+self.out_name+"_scalCovCls.dat"
        clarr = np.loadtxt(filename)
        ells = clarr[:,0]
        ncomps = int(np.sqrt(clarr.shape[1]-1))
        assert ncomps**2 == (clarr.shape[1]-1)
        cls = np.swapaxes(clarr[:,1:],0,1)
        return ells, cls.reshape((ncomps,ncomps,ells.size))
        

    def _replace(self,file_path, pattern, subst):
        # Internal function
        flag = False
        fh, abs_path = mkstemp()
        with open(abs_path,'w') as new_file:
            with open(file_path) as old_file:
                for line in old_file:
                    #if pattern in line:
                    if "".join(line.split())[:len(pattern)+1]==(pattern+"="):
                        line = subst+"\n"
                        flag = True
                    new_file.write(line)
                if not(flag) and ('transfer_redshift' in pattern):
                    line = subst+"\n"
                    flag = True
                    new_file.write(line)
                                                                                   
            if not(flag):
                line = "\n"+subst+"\n"
                new_file.write(line)
                    
        close(fh)
        remove(file_path)
        move(abs_path, file_path)

    def __del__(self):
        remove(self.ifile)        

def test():
    # Demo
    citest = CAMBInterface("params_test.ini",'.')
    citest.set_param("num_redshiftwindows","3")
    citest.set_param("redshift(3)","2")
    citest.set_param("redshift_kind(3)","lensing")
    citest.set_param("redshift_sigma(3)","0.03")
    citest.call(suppress=False)
    ells,cls = citest.get_cls()
    print(cls.shape)
>>>>>>> 643f81438c2ce00084113b37db154d207b0f4963
