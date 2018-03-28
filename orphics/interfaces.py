import os,sys
from tempfile import mkstemp
from shutil import move,copyfile
from os import remove, close
import subprocess
import numpy as np

#TODO:
# test that different MPI jobs get different temp ini files

class CAMBInterface(object):


    def __init__(self,ini_template,camb_loc):

        # cp ini_template to temporary
        self.ifile = ini_template.strip()[:-4]+"_itemp_"+str(os.geteuid())+".ini"
        copyfile(ini_template,self.ifile)
        self.out_name = "itemp_"+str(os.geteuid())
        self.set_param('output_root',self.out_name)

        self.camb_loc = camb_loc

    def set_param(self,param,value):
        self._replace(self.ifile,param,subst=param+"="+str(value))
        
    def call(self,suppress=True):
        if suppress:
            with open(os.devnull, "w") as f:
                subprocess.call([self.camb_loc+"/camb",self.ifile],stdout=f,cwd=self.camb_loc)
        else:
            subprocess.call([self.camb_loc+"/camb",self.ifile],cwd=self.camb_loc)


    def get_cls(self):
        filename =self.camb_loc+"/"+self.out_name+"_scalCovCls.dat"
        clarr = np.loadtxt(filename)
        ells = clarr[:,0]
        ncomps = int(np.sqrt(clarr.shape[1]-1))
        assert ncomps**2 == (clarr.shape[1]-1)
        cls = np.swapaxes(clarr[:,1:],0,1)
        return cls.reshape((ncomps,ncomps,ells.size))
        

    def _replace(self,file_path, pattern, subst):

        flag = False
        fh, abs_path = mkstemp()
        with open(abs_path,'w') as new_file:
            with open(file_path) as old_file:
                for line in old_file:
                    #if pattern in line:
                    if "".join(line.split())[:len(pattern)+1]==(pattern+"="):
                        #print "".join(line.split())[:len(pattern)+1]
                        line = subst+"\n"
                        flag = True
                    new_file.write(line)
                if not(flag) and ('transfer_redshift' in pattern):
                    #print pattern
                    line = subst+"\n"
                    flag = True
                    #print line
                    new_file.write(line)
                                                                                   
            if not(flag):
                line = "\n"+subst+"\n"
                new_file.write(line)
                    
        close(fh)
        remove(file_path)
        move(abs_path, file_path)


        

def test():
    citest = CAMBInterface("params_test.ini",'.')
    citest.set_param("num_redshiftwindows","3")
    citest.set_param("redshift(3)","2")
    citest.set_param("redshift_kind(3)","lensing")
    citest.set_param("redshift_sigma(3)","0.03")
    citest.call(suppress=False)
    cls = citest.get_cls()
    print(cls.shape)
