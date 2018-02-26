"""
Pure E/B for flat-sky from Thubaut Louis'
flipperPol adapted to enlib
"""

import numpy as np
from enlib.fft import fft


class Purify(object):

    def __init__(self,shape,wcs,window):
        px = maps.resolution(shape,wcs)
        self.windict = init_deriv_window(taper,px)
        lxMap,lyMap,self.modlmap,self.angLMap,lx,ly = get_ft_attributes_enmap(shape,wcs)

    def lteb_from_iqu(imap,method='pure'):
        fT, fE, fB = iqu_to_pure_lteb(imap[0],imap[1],imap[2],self.modlmap,self.angLMap,windowDict=self.windict,method=method)
        
        



def init_deriv_window(window,px):
    """
    px is in radians
    """
	
    def matrixShift(l,row_shift,column_shift):	
        m1=np.hstack((l[:,row_shift:],l[:,:row_shift]))
        m2=np.vstack((m1[column_shift:],m1[:column_shift]))
        return m2
    delta=px
    Win=window[:]
    
    dWin_dx=(-matrixShift(Win,-2,0)+8*matrixShift(Win,-1,0)-8*matrixShift(Win,1,0)+matrixShift(Win,2,0))/(12*delta)
    dWin_dy=(-matrixShift(Win,0,-2)+8*matrixShift(Win,0,-1)-8*matrixShift(Win,0,1)+matrixShift(Win,0,2))/(12*delta)
    d2Win_dx2=(-matrixShift(dWin_dx,-2,0)+8*matrixShift(dWin_dx,-1,0)-8*matrixShift(dWin_dx,1,0)+matrixShift(dWin_dx,2,0))/(12*delta)
    d2Win_dy2=(-matrixShift(dWin_dy,0,-2)+8*matrixShift(dWin_dy,0,-1)-8*matrixShift(dWin_dy,0,1)+matrixShift(dWin_dy,0,2))/(12*delta)
    d2Win_dxdy=(-matrixShift(dWin_dy,-2,0)+8*matrixShift(dWin_dy,-1,0)-8*matrixShift(dWin_dy,1,0)+matrixShift(dWin_dy,2,0))/(12*delta)
    
    #In return we change the sign of the simple gradient in order to agree with np convention
    return {'Win':Win, 'dWin_dx':-dWin_dx,'dWin_dy':-dWin_dy, 'd2Win_dx2':d2Win_dx2, 'd2Win_dy2':d2Win_dy2,'d2Win_dxdy':d2Win_dxdy}
	



def iqu_to_pure_lteb(T_map,Q_map,U_map,modLMap,angLMap,windowDict,method='pure'):

    window = windowDict

    win =window['Win']
    dWin_dx=window['dWin_dx']
    dWin_dy=window['dWin_dy']
    d2Win_dx2=window['d2Win_dx2'] 
    d2Win_dy2=window['d2Win_dy2']
    d2Win_dxdy=window['d2Win_dxdy']

    T_temp=T_map.copy()*win
    fT=fft(T_temp,axes=[-2,-1])
    
    Q_temp=Q_map.copy()*win
    fQ=fft(Q_temp,axes=[-2,-1])
    
    U_temp=U_map.copy()*win
    fU=fft(U_temp,axes=[-2,-1])
    
    fE=fT.copy()
    fB=fT.copy()
    
    fE=fQ[:]*np.cos(2.*angLMap)+fU[:]*np.sin(2.*angLMap)
    fB=-fQ[:]*np.sin(2.*angLMap)+fU[:]*np.cos(2.*angLMap)
    
    if method=='standard':
        return fT, fE, fB
    
    Q_temp=Q_map.copy()*dWin_dx
    QWx=fft(Q_temp,axes=[-2,-1])
    
    Q_temp=Q_map.copy()*dWin_dy
    QWy=fft(Q_temp,axes=[-2,-1])
    
    U_temp=U_map.copy()*dWin_dx
    UWx=fft(U_temp,axes=[-2,-1])
    
    U_temp=U_map.copy()*dWin_dy
    UWy=fft(U_temp,axes=[-2,-1])
    
    U_temp=2.*Q_map*d2Win_dxdy-U_map*(d2Win_dx2-d2Win_dy2)
    QU_B=fft(U_temp,axes=[-2,-1])
 
    U_temp=-Q_map*(d2Win_dx2-d2Win_dy2)-2.*U_map*d2Win_dxdy
    QU_E=fft(U_temp,axes=[-2,-1])
    
    modLMap=modLMap+2


    fB[:] += QU_B[:]*(1./modLMap)**2
    fB[:]-= (2.*1j)/modLMap*(np.sin(angLMap)*(QWx[:]+UWy[:])+np.cos(angLMap)*(QWy[:]-UWx[:]))
    
    if method=='hybrid':
        return fT, fE, fB
    
    fE[:]+= QU_E[:]*(1./modLMap)**2
    fE[:]-= (2.*1j)/modLMap*(np.sin(angLMap)*(QWy[:]-UWx[:])-np.cos(angLMap)*(QWx[:]+UWy[:]))
    
    if method=='pure':
        return fT, fE, fB
