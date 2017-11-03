'''
Makes either healpix or flat liteMaps from a shear or spectroscopic catalog.
Run as "python makeMap.py <path to ini file>"
Sample ini file in input/cat.ini
'''


import sys
import os
import time

import configparser
import json

import ctypes
import numpy as np

import flipper.liteMap as lm
import healpy as hp

from scipy.interpolate import interp1d


class fitsMapFromCat(object):
    '''
    A class that generates flat FITS maps from a catalog
    '''

    def __init__(self,suffixes,listOfTemplateLiteMaps,listOfFieldNames):

        self.suffixes = suffixes
        self.tList = listOfTemplateLiteMaps
        self.lMaps = {}
        self.fieldNames = listOfFieldNames
        self.cuts = []

    def _name(self,fieldName,suffix,cutName):
        return fieldName+"_"+suffix+"_"+cutName

    def addCut(self,cutName):

        self.cuts.append(cutName)
        for suffix in self.suffixes:
            for fName,templateA in zip(self.fieldNames,self.tList):
                template = templateA.copy()
                template.data = template.data*0.
                self.lMaps[self._name(fName,suffix,cutName)] = template.copy()
                

    def _isInside(self,template,ra,dec):
        x,y = template.skyToPix(ra,dec)
        return x>=0 and y>=0 and x<template.data.shape[1] and y<template.data.shape[0]


    def getFieldName(self,ra,dec):

        s = self.suffixes[0]
        retFields = []
        for fName in self.fieldNames:
            lmap = self.lMaps[self._name(fName,s,self.cuts[0])]
            if self._isInside(lmap,ra,dec):
                retFields.append(fName)

        return retFields
            


    def updateIndex(self,ra,dec):

        self.fields = self.getFieldName(ra,dec)
        self.indices = []
        for fName in self.fields:
            self.indices.append (self.lMaps[self._name(fName,self.suffixes[0],self.cuts[0])].skyToPix(ra,dec) )

            
    def increment(self,suffix,cutName,incVal):

        for fName,index in zip(self.fields,self.indices):
            self.lMaps[self._name(fName,suffix,cutName)].data[int(index[1]),int(index[0])] += incVal

    def getMap(self,suffix,fieldName,cutName):
        return self.lMaps[self._name(fieldName,suffix,cutName)]

    def writeMaps(self,outputRoot):

        for key in self.lMaps:
            self.lMaps[key].writeFits(outputRoot+key+".fits",overWrite=True)
            print(("Saved ",outputRoot+key+".fits ",self.lMaps[key].data.sum()/(self.lMaps[key].area*60.*60.)," count / arcmin^2"))

        


class healpixFromCat(object):
    '''
    A class that helps with generating healpix maps from a catalog
    '''

    def __init__(self,suffixes,nside=None,deg2hp=None,coord='galactic'):
        # suffixes that specify it's e1,e2,wt, etc.

        assert coord=='galactic' or coord=='equatorial'
        self.coord = coord

        self.suffixes = suffixes
        

        self.npix = hp.pixelfunc.nside2npix(nside)
        self.cnside = ctypes.c_long(nside)
        # this will hold the healpix maps
        self.hp = {}
        self.deg2hp = deg2hp

    def _name(self,suffix,cutName):
        return "hp_"+suffix+"_"+cutName

    def getMap(self,suffix,cutName):
        return self.hp[self._name(suffix,cutName)]


    def addCut(self,cutName):
        
        self._cutName = cutName
        
        for suffix in self.suffixes:
            self.hp[self._name(suffix,cutName)] = np.zeros((self.npix))
            
            

    def updateIndex(self,ra,dec):

        if self.coord=='galactic':
            self.index = self.deg2hp.getPixIndexGalactic(self.cnside,ctypes.c_double(ra),ctypes.c_double(dec))
        elif self.coord=='equatorial':
            self.index = self.deg2hp.getPixIndexEquatorial(self.cnside,ctypes.c_double(ra),ctypes.c_double(dec))

            
    def increment(self,suffix,cutName,incVal):

        self.hp[self._name(suffix,cutName)][self.index] += incVal
                
    def writeMaps(self,outputRoot):

        for key in self.hp:
            hp.write_map(outputRoot+key+".fits",self.hp[key])
            print(("Saved ", outputRoot+key+".fits",self.hp[key].sum()/(0.2*42000.*60.*60.)))

        


class Cut:
    '''
    Define a magnitude and redshift cut.
    Check if (mag,z) satisfies it.
    '''
    def __init__(self,magRange,zRange):


        if magRange==None:
            self.magMin = -np.inf
            self.magMax = np.inf
            magPart = "noMagCut"
        else:
            self.magMin, self.magMax = magRange
            magPart = "magFrom"+str(self.magMin)+"To"+str(self.magMax)

        if zRange==None:
            self.zMin = -np.inf
            self.zMax = np.inf
            zPart = "noZCut"
        else:
            self.zMin, self.zMax = zRange
            zPart = "_zFrom"+str(self.zMin)+"To"+str(self.zMax)

        self.name = magPart + zPart

    def __call__(self,mag,z):
        if (mag<self.magMax and mag>=self.magMin) or mag==None:
            if (z<self.zMax and z>=self.zMin) or z==None:
                return True
            else:
                return False
        else:
            return False    


                
def main(argv):

    # get path to root folder
    try:
        rootPath = os.environ['SKYCROSS_ROOT']
    except KeyError:
        rootPath = ""


        
    # get ini file from command line and load parameters
    if argv==[]:
        print("Using default ini...")
        iniPath = "input/cat.ini"
    else:
        iniFile = argv[0]
        iniPath = rootPath + iniFile
    Config = configparser.SafeConfigParser()
    Config.read(iniPath)

    mapType = Config.get('general','type')
    isHp = Config.getboolean('general','healpix')
    catPaths = Config.get('general','cat_file').split(',')
    skipNum = Config.getint('general','skip_rows')
    maxNum = Config.getint('general','max_rows')
    delim = Config.get('general','delimiter').replace("'", "")
    outputRoot = Config.get('general','output_root')

    skip_char = Config.get('general','skip_char')

    try:
        isMock = Config.getboolean('columns','mock')
    except:
        isMock = False    

    




    # print skip_char
    # print catPaths
    # sys.exit()


    cNum = Config._sections['columns']
    for key in cNum:
        if key=='__name__': continue
        try:
            cNum[key] = int(cNum[key])
        except ValueError:
            cNum.pop(key,None)

    try:        
        magRanges = json.loads(Config.get('general','mag_ranges'))
    except ValueError:
        print("No magnitude cut specified / unreadable magnitude cuts. No mag cuts will be made.")
        magRanges = [None]
    try:        
        zRanges = json.loads(Config.get('general','z_ranges'))
    except ValueError:
        print("No z cut specified / unreadable z cuts. No z cuts will be made.")
        zRanges = [None]
    
            
    # initialize liteMaps if necessary
    if not(isHp):
        mBounds = json.loads(Config.get('flat','fieldBounds'))
        fNames = json.loads(Config.get('flat','fieldNames'))
        
        px = Config.getfloat('flat','pixScaleArcmin')
        
    # initialize healpix maps
    else:
        # Set up fast C library for ra/dec to healpix index conversion
        deg2pix = ctypes.CDLL(rootPath + 'lib/deg2healpix.so')
        deg2pix.getPixIndexGalactic.argtypes = (ctypes.c_long, ctypes.c_double, ctypes.c_double)
        deg2pix.getPixIndexEquatorial.argtypes = (ctypes.c_long, ctypes.c_double, ctypes.c_double)
        nside = Config.getint('healpix','nside')


        
    try:        
        zCol = cNum['z']
        haveZ = True
    except KeyError:
        haveZ = False    



    fcwts = lambda x: 1.
    if haveZ:
        try:
            czs, cwts = np.loadtxt(Config.get('general','cfhtlens_weights'),delimiter=' ',unpack = True)
            fcwts = interp1d(czs,cwts,bounds_error=False,fill_value=1.)
            print(("Using cfht weights from file ", Config.get('general','cfhtlens_weights')))
        except:
            pass

        
    try:        
        magCol = cNum['mag']
        haveMag = True
        #validateMagRanges()
    except KeyError:
        haveMag = False    
        


    raCol = cNum['ra']
    decCol = cNum['dec']

    '''
    For a shear catalog, we need to prepare the following maps:
    1. (e1-c1)*w
    2. (e2-c2)*w
    3. (1+m)*w
    4. w
    5. counts

    For a spec catalog,
    1. counts
    2. counts*w


    '''

    try:
        wCol = cNum['w']
        haveW = True
    except KeyError:
        haveW = False    

    
    if "shear" in mapType:
        suffixes = ['_e1','_e2','_mw','_w','_ct']
    elif mapType.startswith("spec"):
        suffixes = ['_ct','_wct']
    
    myCuts = {}

    if isHp:
        mapSets = genMapSet(suffixes,nside=nside,deg2hp=deg2pix)
    else:
        mapSets = genMapSet(suffixes,fieldBounds=mBounds,fieldNames=fNames,pxScaleArcmin=px)
    
    if magRanges==[]: magRanges=[None]
    if zRanges==[]: zRanges=[None]


    
    for magRange in magRanges:
        try:
            if magRange.upper()=="NONE": magRange = None
        except AttributeError:
            pass        
        for zRange in zRanges:
            try:
                if zRange.upper()=="NONE": zRange = None
            except AttributeError:
                pass        
            newCut = Cut(magRange,zRange)
            myCuts[newCut.name] = newCut
            mapSets.addCut(newCut.name)


    #sys.exit()
    
    i=0
    z = None
    mag = None

    start_time=time.time()
    # ras = []
    # decs = []

    for catPath in catPaths:
        with open(catPath) as f:
            for line in f:
                i+=1
                if i<=skipNum: continue # verify
                if line.lstrip()[0] == skip_char: continue
                if i>maxNum: break
                if i%50000==0: print(i)

                try:
                    cols = line.split(delim.decode('string_escape'))
                except:
                    cols = line.split()

                ra = float(cols[raCol])
                dec = float(cols[decCol])

                # ras.append(ra)
                # decs.append(dec)
                # continue

                if haveMag:
                    mag = float(cols[magCol])


                if haveZ:
                    z = float(cols[zCol])
                    if z<0.: continue

                mapSets.updateIndex(ra,dec)
                if not(isHp):
                    if mapSets.fieldName==None: continue

                for cutName,myCut in list(myCuts.items()):

                    if not(myCut(mag,z)): continue


                    if mapType=='shear':
                        e1 = float(cols[cNum['e1']])
                        e2 = float(cols[cNum['e2']])
                        mcorr = float(cols[cNum['mcorr']])
                        w = float(cols[wCol])
                        try:
                            c1 = float(cols[cNum['c1']])
                        except KeyError:
                            c1 = 0.    
                        try:
                            c2 = float(cols[cNum['c2']])
                        except KeyError:
                            c2 = 0.    

                        e1Effective = w*(e1 - c1)
                        e2Effective = w*(e2 - c2)   

                        mapSets.increment("_e1",cutName,e1Effective)
                        mapSets.increment("_e2",cutName,e2Effective)
                        mapSets.increment("_mw",cutName,w*(1.+mcorr))
                        mapSets.increment("_w",cutName,w)
                        mapSets.increment("_ct",cutName,1.0)

                    elif mapType=='hsc_shear':
                        e1 = float(cols[cNum['e1']])
                        e2 = float(cols[cNum['e2']])
                        sigma = float(cols[wCol])

                        rfact = 0.365
                        R = 1 - rfact**2.
                        w = 1./(rfact**2. + sigma**2.)
                        
                        e1Effective = w*e1/R
                        e2Effective = w*e2/R  

                        mapSets.increment("_e1",cutName,e1Effective)
                        mapSets.increment("_e2",cutName,e2Effective)
                        mapSets.increment("_w",cutName,w)
                        mapSets.increment("_ct",cutName,1.0)

                    elif mapType.startswith("spec"):                    

                        mapSets.increment("_ct",cutName,1.0)

                        if isMock:
                            #pthalos
                            # wt_boss = int(cols[4])
                            # wt_cp = int(cols[5])
                            # wt_red = int(cols[6])
                            # veto = int(cols[7])
                            # if veto!=1: continue
                            # if wt_boss<=0 or wt_cp<=0 or wt_red<=0: continue
                            # w = float(wt_cp+wt_red-1.) * fcwts(z)
                            # mapSets.increment("_wct",cutName,w)

                            #patchy
                            veto = int(cols[6])
                            wt_cp = int(cols[7])
                            if veto!=1: continue
                            if wt_cp<=0 : continue
                            w = float(wt_cp) * fcwts(z)
                            mapSets.increment("_wct",cutName,w)
                                                        
                            
                        elif haveW:
                            w = float(cols[wCol]) * fcwts(z)
                            mapSets.increment("_wct",cutName,w)
                

    # print min(ras),max(ras)
    # print min(decs),max(decs)
    # sys.exit()
    stop_time=time.time()
    N = Config.getint('debug','estimate_rows')
    print(i)
    avg_time = (stop_time-start_time)/i
    print((avg_time * N / 60. , " minutes expected."))
                        
    mapSets.writeMaps(outputRoot+mapType+"_")

if (__name__ == "__main__"):
    main(sys.argv[1:])


            
