import os
import csv
import time
import sys
import subprocess

import atexit
import pickle as pickle
import traceback
from functools import wraps
import numpy as np

from tempfile import mkstemp
from shutil import move,copyfile
from os import remove, close

from scipy.interpolate import interp1d

# Interface code to call CAMB (not pycamb) with different settings
# This code includes the following options:
#     0: original CAMB
#     1: CAMB sources

class cambInterface:
    def __init__(self,outName,templateIni,cambRoot="/gpfs/home/nhnguyen/software/cambRedNN",option=0,seed=0):
        self._template = cambRoot+"/cambInterface_temp_"+str(seed)+".ini"
        copyfile(templateIni, self._template)
        self._callpath = cambRoot
        self.setParam('output_root',outName)
        self.outName = outName
        self.option = option
        if option == 0:
            print('Using original CAMB')
        elif option == 1:
            print('Using CAMB sources')
        
    def setParam(self,paramName,newVal):
        self._replace(self._template,paramName,subst=paramName+"="+str(newVal))

    def call(self,suppress=True):
        print((self._template))
        if suppress:
            with open(os.devnull, "w") as f:
                subprocess.call([self._callpath+"/camb",self._template],stdout=f,cwd=self._callpath)
        else:
            subprocess.call([self._callpath+"/camb",self._template],cwd=self._callpath)

    def done(self):
        remove(self._template)

    def getCls(self,lensed=False):#,lpower=1,fact=2.*np.pi):
        
        def _map(ells,clls,ellevals):
            clfunc = interp1d(ells.ravel(),clls.ravel(),bounds_error=False,fill_value=0.)
            return clfunc(ellevals)

        if self.option == 0:
            filename =self._callpath+"/"+self.outName+"_lenspotentialCls.dat"
            clarr = np.loadtxt(filename)
            ells = clarr[:,0].ravel()
            elleval = np.arange(0,max(ells),1)
            
            
            clkk = clarr[:,5].ravel() *(2.*np.pi/4.)
            clkkeval = _map(ells,clkk,elleval)
            clkt = clarr[:,6].ravel() *(2.*np.pi/2.) / np.sqrt(ells*(ells+1.))
            clkteval = _map(ells,clkt,elleval)
            
            
            if lensed:
                filename =self._callpath+"/"+self.outName+"_lensedCls.dat"
            else:
                filename =self._callpath+"/"+self.outName+"_lenspotentialCls.dat"
            
            clarr = np.loadtxt(filename)
            ells = clarr[:,0].ravel()
            ellevalcmb = np.arange(0,max(ells),1)
            
            N = min(len(elleval),len(ellevalcmb))
        
            clvecs = []
            for i in range(1,5):
                clcmb = clarr[:,i].ravel() *2.*np.pi/ (ells*(ells+1.))
                clcmbeval = _map(ells,clcmb,ellevalcmb)[:N].reshape((N,1))
                clvecs.append(clcmbeval)
                
                if i==1: print((clcmbeval[:10]))
            
            clkkeval = clkkeval[:N]
            clkteval = clkteval[:N]
            retCls = np.hstack(tuple(clvecs+[clkkeval.reshape((N,1)),clkteval.reshape((N,1))]))
            
            return retCls
                        
        elif self.option == 1:
            '''
            Use cambRed for galaxy
            Return Clkk,Clkg1,...,Clkgn,Clg1g1,...,Clg1gn,Clg2g1,...,Clgngn
            '''
            
            filename =self._callpath+"/"+self.outName+"_scalCovCls.dat"
            clarr = np.loadtxt(filename)
            ells = clarr[:,0].ravel()
            elleval = np.arange(0,max(ells),1)
            N = len(elleval)

            # nxn matrix
            n = int(np.sqrt(len(clarr[0,:])-1))
            # Get kk
            clkk = clarr[:,(n*2+3)].ravel()*(2.*np.pi/4.)*(ells*(ells+1.))
            clkkeval = _map(ells,clkk,elleval)[:N]
            Cls = clkkeval.reshape(N,1)
            
            winNum = n-3
            # Get kg
            for i in range(winNum):
                clkg = clarr[:,(n*2+4+i)].ravel() *(2.*np.pi/2.)
                clkgeval = _map(ells,clkg,elleval)[:N]
                Cls = np.hstack([Cls,clkgeval.reshape(N,1)])
            for i in range(winNum):
                for j in range(winNum-i):
                    clgg = clarr[:,(n*(3+i)+4+i+j)].ravel()*(2.*np.pi)/(ells*(ells+1.))
                    clggeval = _map(ells,clgg,elleval)[:N]
                    Cls = np.hstack([Cls,clggeval.reshape(N,1)])
            
            #clkt = clarr[:,3].ravel() *(2.*np.pi/2.)
            #clkteval = _map(ells,clkt,elleval)
            #retCls = np.hstack(tuple(clvecs+[clkkeval.reshape((N,1)),clkteval.reshape((N,1))]))
            return np.nan_to_num(Cls)

        #elif self.option == 2:
       

    def _replace(self,file_path, pattern, subst):

        flag = False
        fh, abs_path = mkstemp()
        with open(abs_path,'w') as new_file:
            with open(file_path) as old_file:
                for line in old_file:
                    if "".join(line.split())[:len(pattern)+1]==(pattern+"="):
                        line = subst+"\n"
                        flag = True
                    new_file.write(line)
                if not(flag) and ('transfer_redshift' in pattern):
                    line = subst+"\n"
                    flag = True
                    #print line
                    new_file.write(line)
                                                                                   
                    
        if not(flag):

            line = "\n"+pattern+"="+ subst+"\n"
            new_file.write(line)
            
        close(fh)
        remove(file_path)
        move(abs_path, file_path)


