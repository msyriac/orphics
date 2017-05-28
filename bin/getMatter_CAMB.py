import numpy as np
import sys,os
import ConfigParser
import itertools
import os.path
from orphics.theory.cambCall import cambInterface

'''
This script gets the matter power spectrum P(k) from CAMB, axionCAMB, ... etc.
Modify the ini file to run. The code returns the matter file name.
'''

def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)

def getMatterCamb(out_pre,spec,templateIni,params,zpts,zmin,zmax,cambRoot,AccuracyBoost=False,seed=0):
    option = 0 # for regular CAMB call
    CAMB = cambInterface(out_pre,templateIni,cambRoot=cambRoot,option=option,seed=seed)
    for key in params:
        CAMB.setParam(key,params[key])
    CAMB.call(suppress=False)
    CAMB.done()

    ### Merge data from all zs, remove all matterpower and transfer dat files
    name = cambRoot+out_pre+'_matterpower'
    nameTransfer = cambRoot+out_pre+'_transfer'

    # First file name is different
    data = np.loadtxt(name+'.dat')
    os.remove(name+'.dat')
    os.remove(nameTransfer+'_out.dat')

    N = len(data)
    ks = data[:,0].reshape(N,1)
    data = data[:,1].reshape(N,1)

    # Looping over the rest
    for i in range(2,zpts+1):
        PS = np.loadtxt(name+'_'+str(i)+'.dat')
        data = np.hstack([data,PS[:,1].reshape(N,1)])
        os.remove(name+'_'+str(i)+'.dat')
        os.remove(nameTransfer+'_'+str(i)+'.dat')
    fileName = name+'_all.dat'
    # Flip the columns since CAMB calculate higher z first
    data = np.hstack([ks,data[:,::-1]])
    np.savetxt(fileName,data)
    # Add z range as a comment on the first line
    line = '####\t'+'\t'.join([str(x) for x in np.linspace(zmin,zmax,zpts)])
    line_prepender(fileName,line)

    return fileName

def main(argv):
    verbose=True
    nameMap={'H0':'hubble','YHe':'helium_fraction','nnu':'massless_neutrinos','s_pivot':'pivot_scalar','t_pivot':'pivot_tensor','As':'scalar_amp(1)','ns':'scalar_spectral_index(1)','tau':'re_optical_depth','num_massive_neutrinos':'massive_neutrinos','TCMB':'temp_cmb','lmax':'l_max_scalar','kmax':'k_eta_max_scalar'}
    inv_nameMap = {v: k for k, v in nameMap.items()}

    try:
        iniFile = argv[0]
    except:
        iniFile = "input/getMatter_axionCAMB.ini"
        
    # Read Config
    Config = ConfigParser.SafeConfigParser()
    Config.optionxform = str
    Config.read(iniFile)
    
    out_pre = Config.get('general','output_prefix')
    spec = Config.get('general','spec')
    AccuracyBoost = Config.getboolean('general','AccuracyBoost')
    cambRoot = Config.get('general','cambRoot')
    templateIni = Config.get('general','templateIni')
    if not os.path.isfile(templateIni):
        templateIni = cambRoot+Config.get('general','templateIni')
            
    zmin,zmax,zpts = [float(x) for x in Config.get('general','redshifts').split(',')]
    zrange = np.linspace(zmin,zmax,int(zpts))
    
    paramList = []
    fparams = {}
    
    for (key, val) in Config.items('CAMB'):
        if key in nameMap:
            key = nameMap[key]
        try:
            fparams[key] = eval(val)
        except:
            fparams[key] = val
        '''
        if key in ['l_max_scalar','massive_neutrinos','do_nonlinear','halofit_version']:
            fparams[key] = int(val)
        else:
            try:
                fparams[key] = float(val)
            except:
                fparams[key] = val
        '''
    if 'massless_neutrinos' in fparams:
        fparams['massless_neutrinos'] = fparams['massless_neutrinos']-fparams['massive_neutrinos']
    if not('omnuh2' in fparams) and ('mnu' in fparams):
        fparams['omnuh2'] = round(fparams['mnu']/93.14,6)
        fparams.pop('mnu')
    fparams['transfer_num_redshifts']  = int(zpts)
    for j in range(int(zpts)):
        #CAMB evaluates higher z first
        fparams['transfer_redshift('+str(j+1)+')'] = zrange[::-1][j]
    
    #print fparams
    Pk = getMatterCamb(out_pre,spec,templateIni,fparams,AccuracyBoost=AccuracyBoost,zpts=int(zpts),zmin=zmin,zmax=zmax,cambRoot=cambRoot)

    print Pk
if (__name__ == "__main__"):
    main(sys.argv[1:])
