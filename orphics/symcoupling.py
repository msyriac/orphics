from __future__ import print_function
import numpy as np
from sympy import Symbol,Function
import sympy
from pixell import fft as efft, enmap
from orphics import maps,io,stats,cosmology,lensing
import os,sys

"""
Routines to reduce and evaluate symbolic mode coupling integrals
"""

ifft = lambda x: efft.ifft(x,axes=[-2,-1],normalize=True)
fft = lambda x: efft.fft(x,axes=[-2,-1])


def factorize_2d_convolution_integral(expr,l1funcs=None,l2funcs=None,groups=None,validate=True):
    """Reduce a sympy expression of variables l1x,l1y,l2x,l2y,l1,l2 into a sum of 
    products of factors that depend only on vec(l1) and vec(l2) and neither, each. If the expression
    appeared as the integrand in an integral over vec(l1), where 
    vec(l2) = vec(L) - vec(l1) then this reduction allows one to evaluate the 
    integral as a function of vec(L) using FFTs instead of as a convolution.
    """

    # Generic message if validation fails
    val_fail_message = "Validation failed. This expression is likely not reducible to FFT form."
    # Get the 2D convolution cartesian variables
    l1x,l1y,l2x,l2y,l1,l2 = get_ells()
    if l1funcs is None: l1funcs = []
    if l2funcs is None: l2funcs = []
    if l1x not in l1funcs: l1funcs.append(l1x)
    if l1y not in l1funcs: l1funcs.append(l1y) 
    if l1 not in l1funcs: l1funcs.append(l1)
    if l2x not in l2funcs: l2funcs.append(l2x)
    if l2y not in l2funcs: l2funcs.append(l2y) 
    if l2 not in l2funcs: l2funcs.append(l2)
    Lx = Symbol('Lx')
    Ly = Symbol('Ly')
    L = Symbol('L')
    ofuncs1 = set(l1funcs) - set([l1x,l1y,l1])
    ofuncs2 = set(l2funcs) - set([l2x,l2y,l2])

   
    # List to collect terms in
    terms = []
    if validate: prodterms = []
    # We must expand the expression so that the top-level operation is Add, i.e. it looks like
    # A + B + C + ...
    expr = sympy.expand( expr )
    # What is the top-level operation?
    op = expr.func
    if op is sympy.Add:
        arguments = expr.args # If Add, then we have multiple terms
    else:
        arguments = [expr] # If not Add, then we have a single term
    # Let's factorize each term
    unique_l1s = []
    unique_l2s = []
    
    def homogenize(inexp):
        outexp = inexp.subs([[l1x,Lx],[l2x,Lx],[l1y,Ly],[l2y,Ly],[l1,L],[l2,L]])
        ofuncs = ofuncs1.union(ofuncs2)
        for ofunc in ofuncs:
            nfunc = Symbol(str(ofunc)[:-3])
            outexp = outexp.subs(ofunc,nfunc)
        return outexp

    
    def get_group(inexp):
        if groups is None: return 0
        found = False
        d = Symbol('dummy')
        for i,group in enumerate(groups):
            s = inexp.subs(group,d)
            if not((s/d).has(d)):
                if found:
                    print(s,group)
                    raise ValueError("Groups don't seem to be mutually exclusive.")
                index = i
                found = True
        if not(found):
            raise ValueError("Couldn't associate a group")
        return index


    ogroups = [] if not(groups is None) else None
    ogroup_weights = [] if not(groups is None) else None
    ogroup_symbols = sympy.ones(len(groups),1) if not(groups is None) else None
    for k,arg in enumerate(arguments):
        temp, ll1terms = arg.as_independent(*l1funcs, as_Mul=True)
        loterms, ll2terms = temp.as_independent(*l2funcs, as_Mul=True)

        
        if any([x==0 for x in [ll1terms,ll2terms,loterms]]): continue
        # Group ffts
        if groups is not None:
            gindex = get_group(loterms)
            ogroups.append(gindex)
            fsyms = loterms.free_symbols
            ocoeff = loterms.evalf(subs=dict(zip(fsyms,[1]*len(fsyms))))
            ogroup_weights.append( float(ocoeff) )
            if ogroup_symbols[gindex]==1:
                ogroup_symbols[gindex] = loterms/ocoeff
            else:
                assert ogroup_symbols[gindex]==loterms/ocoeff, "Error validating group membership"
        
        vdict = {}
        vdict['l1'] = ll1terms
        vdict['l2'] = ll2terms
        tdict = {}
        tdict['l1'] = homogenize(vdict['l1'])
        tdict['l2'] = homogenize(vdict['l2'])

        if not(tdict['l1'] in unique_l1s):
            unique_l1s.append(tdict['l1'])
        tdict['l1index'] = unique_l1s.index(tdict['l1'])

        if not(tdict['l2'] in unique_l2s):
            unique_l2s.append(tdict['l2'])
        tdict['l2index'] = unique_l2s.index(tdict['l2'])

        
        
        vdict['other'] = loterms
        tdict['other'] = loterms
        terms.append(tdict)
        # Validate!
        if validate:
            # Check that all the factors of this term do give back the original term
            products = sympy.Mul(vdict['l1'])*sympy.Mul(vdict['l2'])*sympy.Mul(vdict['other'])
            assert sympy.simplify(products-arg)==0, val_fail_message
            prodterms.append(products)
            # Check that the factors don't include symbols they shouldn't
            assert all([not(vdict['l1'].has(x)) for x in l2funcs]), val_fail_message
            assert all([not(vdict['l2'].has(x)) for x in l1funcs]), val_fail_message
            assert all([not(vdict['other'].has(x)) for x in l1funcs]), val_fail_message
            assert all([not(vdict['other'].has(x)) for x in l2funcs]), val_fail_message
    # Check that the sum of products of final form matches original expression
    if validate:
        fexpr = sympy.Add(*prodterms)
        assert sympy.simplify(expr-fexpr)==0, val_fail_message
    return terms,unique_l1s,unique_l2s,ogroups,ogroup_weights,ogroup_symbols




def get_ells():
    l1x = Symbol('l1x')
    l1y = Symbol('l1y')
    l2x = Symbol('l2x')
    l2y = Symbol('l2y')
    l1 = Symbol('l1')
    l2 = Symbol('l2')
    return l1x,l1y,l2x,l2y,l1,l2

def get_Ls():
    Lx = Symbol('Lx')
    Ly = Symbol('Ly')
    L = Symbol('L')
    return Lx,Ly,L
    

class ModeCoupling(object):

    def __init__(self,shape,wcs,groups=None):
        # Symbolic
        self.l1x,self.l1y,self.l2x,self.l2y,self.l1,self.l2 = get_ells()
        self.Lx,self.Ly,self.L = get_Ls()
        if groups is None: groups = [self.Lx*self.Lx,self.Ly*self.Ly,self.Lx*self.Ly]
        self._default_groups = groups
        self.integrands = {}
        self.ul1s = {}
        self.ul2s = {}
        self.ogroups = {}
        self.ogroup_weights = {}
        self.ogroup_symbols = {}
        self.l1funcs = []
        self.l2funcs = []
        # Diagnostic
        self.nfft = 0
        self.nifft = 0
        # Numeric
        self.shape,self.wcs = shape,wcs
        self.modlmap = enmap.modlmap(shape,wcs)
        self.lymap,self.lxmap = enmap.lmap(shape,wcs)
        self.pixarea = np.prod(enmap.pixshape(shape,wcs))
        

    def f_of_ell(self,name,ell):
        fname = name+"_"+str(ell)
        f = Symbol(fname)
        if ell in [self.l1x,self.l1y,self.l1]: self.l1funcs.append(f)
        if ell in [self.l2x,self.l2y,self.l2]: self.l2funcs.append(f)
        return f
    
    def add_factorized(self,tag,expr,validate=True,groups=None):
        self.integrands[tag],self.ul1s[tag],self.ul2s[tag], \
            self.ogroups[tag],self.ogroup_weights[tag], \
            self.ogroup_symbols[tag] = factorize_2d_convolution_integral(expr,l1funcs=self.l1funcs,l2funcs=self.l2funcs,
                                                                         validate=validate,groups=groups)


    def integrate(self,tag,feed_dict,xmask=None,ymask=None,cache=True,pixel_units=False):
        feed_dict['L'] = self.modlmap
        feed_dict['Ly'] = self.lymap
        feed_dict['Lx'] = self.lxmap
        shape = self.shape[-2:]
        ones = np.ones(shape,dtype=feed_dict['L'].dtype)
        val = 0.
        if xmask is None: xmask = ones
        if ymask is None: ymask = ones
        

        if cache:
            cached_u1s = []
            cached_u2s = []
            for u1 in self.ul1s[tag]:
                l12d = evaluate(u1,feed_dict)*ones
                cached_u1s.append(self._ifft(l12d*xmask))
            for u2 in self.ul2s[tag]:
                l22d = evaluate(u2,feed_dict)*ones
                cached_u2s.append(self._ifft(l22d*ymask))


        # For each term, the index of which group it belongs to  
        ogroups = self.ogroups[tag] 
        ogroup_weights = self.ogroup_weights[tag] 
        ogroup_symbols = self.ogroup_symbols[tag]

        def get_l1l2(term):
            if cache:
                ifft1 = cached_u1s[term['l1index']]
                ifft2 = cached_u2s[term['l2index']]
            else:
                l12d = evaluate(term['l1'],feed_dict)*ones
                ifft1 = self._ifft(l12d*xmask)
                l22d = evaluate(term['l2'],feed_dict)*ones
                ifft2 = self._ifft(l22d*ymask)
            return ifft1,ifft2
        
        
        if ogroups is None:    
            for i,term in enumerate(self.integrands[tag]):
                ifft1,ifft2 = get_l1l2(term)
                ot2d = evaluate(term['other'],feed_dict)*ones
                ffft = self._fft(ifft1*ifft2)
                val += ot2d*ffft
        else:
            vals = np.zeros((len(ogroup_symbols),)+shape,dtype=feed_dict['L'].dtype)+0j
            for i,term in enumerate(self.integrands[tag]):
                ifft1,ifft2 = get_l1l2(term)
                gindex = ogroups[i]
                vals[gindex,...] += ifft1*ifft2 *ogroup_weights[i]
            for i,group in enumerate(ogroup_symbols):
                ot2d = evaluate(ogroup_symbols[i],feed_dict)*ones            
                ffft = self._fft(vals[i,...])
                val += ot2d*ffft

                
        mul = 1 if pixel_units else 1./self.pixarea
        return val * mul

    def _fft(self,x):
        self.nfft += 1
        return fft(x+0j)
    def _ifft(self,x):
        self.nifft += 1
        return ifft(x+0j)
        

class LensingModeCoupling(ModeCoupling):
    def __init__(self,shape,wcs,theory=None,theory_norm=None,lensed_cls=None):
        ModeCoupling.__init__(self,shape,wcs)
        self.Lx,self.Ly,self.L = get_Ls()
        self.Ldl1 = (self.Lx*self.l1x+self.Ly*self.l1y)
        self.Ldl2 = (self.Lx*self.l2x+self.Ly*self.l2y)
        self.l1dl2 = (self.l1x*self.l2x+self.l2x*self.l2y)
        self.cos2t12,self.sin2t12 = substitute_trig(self.l1x,self.l1y,self.l2x,self.l2y,self.l1,self.l2)

        self.theory2d = None
        if theory is not None:
            self.theory2d = self._load_theory(theory,lensed_cls=lensed_cls)
        if theory_norm is None:
            self.theory2d_norm = self.theory2d
        else:
            self.theory2d_norm = self._load_theory(theory_norm,lensed_cls=lensed_cls)

        self.Als = {}
        
    def _load_theory(self,theory,lensed_cls=None):
        pol_list = ['TT','EE','TE','BB']
        cls = {}
        for pol in pol_list:
            cls["u"+pol] = theory.uCl(pol,self.modlmap) if ((lensed_cls is None) or not(lensed_cls)) else theory.lCl(pol,self.modlmap)
            cls["l"+pol] = theory.lCl(pol,self.modlmap) if ((lensed_cls is None) or lensed_cls) else theory.uCl(pol,self.modlmap)
        return cls

    def Cls(self,name,ell,pols=['TT','EE','TE','BB']):
        r = {}
        for p in pols:
            fname = name+"_"+p
            r[p] = self.f_of_ell(fname,ell)
        return r
        
    def f(self,polcomb,uCl1,uCl2,rev=False):
        Ldl1 = self.Ldl1 if not(rev) else self.Ldl2
        Ldl2 = self.Ldl2 if not(rev) else self.Ldl1
        u1 = uCl1 if not(rev) else uCl2
        u2 = uCl2 if not(rev) else uCl1
        
        if polcomb=='TT':
            return Ldl1*u1['TT']+Ldl2*u2['TT']
        elif polcomb=='TE':
            return Ldl1*self.cos2t12*u1['TE']+Ldl2*u2['TE']
        elif polcomb=='ET':
            return Ldl2*u2['TE']*self.cos2t12+Ldl1*u1['TE']
        elif polcomb=='TB':
            return u1['TE']*self.sin2t12*Ldl1
        elif polcomb=='EE':
            return (Ldl1*u1['EE']+Ldl2*u2['EE'])*self.cos2t12
        elif polcomb=='EB':
            return Ldl1*u1['EE']*self.sin2t12

    def F_HuOk(self,polcomb,tCl1,tCl2,uCl1,uCl2,rev=False,f=None):
        t1 = tCl1 if not(rev) else tCl2
        t2 = tCl2 if not(rev) else tCl1
        
        if f is None: f = self.f(polcomb,uCl1,uCl2,rev=rev)
        if polcomb=='TE':
            frev = self.f(polcomb,uCl1,uCl2,rev=not(rev))
            # this filter is not separable
            #(tCl1['EE']*tCl2['TT']*f - tCl1['TE']*tCl2['TE']*frev)/(tCl1['TT']*tCl2['EE']*tCl1['EE']*tCl2['TT']-(tCl1['TE']*tCl2['TE'])**2.)
            # this approximation is
            return (t1['EE']*t2['TT']*f - t1['TE']*t2['TE']*frev)/(t1['TT']*t2['EE']*t1['EE']*t2['TT'])

        # frev = self.f(polcomb,uCl1,uCl2,rev=not(rev))
        # ct1 = self.Cls("uCl",self.l1)
        # ct2 = self.Cls("uCl",self.l2)
        # tp2 = self.Cls("tClX",self.l2)
        # tp1 = self.Cls("tClY",self.l1)
        # tp2p = self.Cls("tClX",self.l1)
        # tp1p = self.Cls("tClY",self.l2)
        
        # return (tp1['TT']*tp2['TT']*f - ct1['TT']*ct2['TT']*frev)/(tp1['TT']*tp2['TT']*tp1p['TT']*tp2p['TT'])

            
        X,Y = polcomb
        pfact = 0.5 if Y!='B' else 1.0
        return f*pfact/t1[X+X]/t2[Y+Y] 
        

        
    def F_HDV(self,polcomb,tCl1,tCl2,uCl1,uCl2=None,rev=False):
        t1 = tCl1 if not(rev) else tCl2
        t2 = tCl2 if not(rev) else tCl1
        u1 = uCl1 if not(rev) else uCl2

        X,Y = polcomb
        Ldl1 = self.Ldl1 if not(rev) else self.Ldl2
        pref = Ldl1/t1[X+X]/t2[Y+Y]
        if Y=='T':
            return pref * u1[X+Y]
        if Y=='E':
            return pref * u1[X+Y] * self.cos2t12
        if Y=='B':
            return pref * u1[X+'E'] * self.sin2t12
            
        
    def add_ALinv(self,tag,fa,Fa,validate=True,groups=None):
        expr = fa*Fa/self.L**2
        self.add_factorized(tag,expr,validate=validate,groups=groups)

    def get_AL(self,tag,feed_dict,xmask=None,ymask=None,cache=True):
        ival = self.integrate(tag,feed_dict,xmask=xmask,ymask=ymask,cache=cache).real
        return np.nan_to_num(1./ival)

    def NL_from_AL(self,AL):
        return np.nan_to_num(AL*self.modlmap**2./4.)
        
    def get_NL(self,tag,feed_dict,xmask=None,ymask=None,cache=True):
        AL = self.get_AL(tag,feed_dict,xmask=xmask,ymask=xmask,cache=cache)
        return self.NL_from_AL(AL)
        
    def add_cross(self,tag,Fa,Fb,Fbr,Cxaxb1,Cyayb2,Cxayb1,Cyaxb2,validate=True,groups=None):
        # integrand in Eq 17 in HuOk01
        if groups is None: groups = self._default_groups
        expr = Fa*(Fb*Cxaxb1*Cyayb2+Fbr*Cxayb1*Cyaxb2)
        self.add_factorized(tag,expr,validate=validate,groups=groups)    

    def AL(self,pol,xmask,ymask,noise_t=0,noise_e=0,noise_b=0,
           ynoise_t=None,ynoise_e=None,ynoise_b=None,
           save_expression="current",
           theory=None,theory_norm=None,
           hdv=True,validate=True,lensed_cls=None,cache=True,groups=None):

        if ynoise_t is None: ynoise_t = noise_t
        if ynoise_e is None: ynoise_e = noise_e
        if ynoise_b is None: ynoise_b = noise_b
        uCl1 = self.Cls("uCl",self.l1)
        uCl2 = self.Cls("uCl",self.l2)
        tCl1 = self.Cls("tClX",self.l1)
        tCl2 = self.Cls("tClY",self.l2)
        uCl1N = self.Cls("uClN",self.l1) if not(theory_norm is None) else uCl1
        uCl2N = self.Cls("uClN",self.l2) if not(theory_norm is None) else uCl2
        f_norm = self.f(pol,uCl1N,uCl2N)
        F = self.F_HDV(pol,tCl1,tCl2,uCl1) if hdv else self.F_HuOk(pol,tCl1,tCl2,uCl1,uCl2)
        if groups is None: groups = self._default_groups
        self.add_ALinv(save_expression,f_norm,F,validate=validate,groups=groups)

        theory2d,theory2d_norm = self._get_theory2d(theory,theory_norm,lensed_cls)
        feed_dict = self._dict_from_theory_noise(theory2d,theory2d_norm,noise_t,ynoise_t,noise_e,ynoise_e,noise_b,ynoise_b)
        return self.get_AL(save_expression,feed_dict,xmask=xmask,ymask=ymask,cache=cache)

    def _get_theory2d(self,theory,theory_norm,lensed_cls):
        if theory is None:
            theory2d = self.theory2d
        else:
            theory2d = self._load_theory(theory,lensed_cls=lensed_cls)
        if theory_norm is None:
            theory2d_norm = theory2d
        else:
            theory2d_norm = self._load_theory(theory_norm,lensed_cls=lensed_cls)
        return theory2d,theory2d_norm

    def dict_from_noise(self,xnoise_t,ynoise_t=None,
                                xnoise_e=None,ynoise_e=None,
                                xnoise_b=None,ynoise_b=None,
                                cxnoise_t=None,cynoise_t=None,
                                cxnoise_e=None,cynoise_e=None,
                                cxnoise_b=None,cynoise_b=None):

        if ynoise_t is None: ynoise_t = xnoise_t
        if xnoise_e is None: xnoise_e = 2*xnoise_t
        if xnoise_b is None: xnoise_b=xnoise_e
        if ynoise_e is None: ynoise_e=2*ynoise_t
        if ynoise_b is None: ynoise_b=ynoise_e
        if cxnoise_t is None: cxnoise_t=xnoise_t
        if cxnoise_e is None: cxnoise_e=xnoise_e
        if cxnoise_b is None: cxnoise_b=xnoise_b
        if cynoise_t is None: cynoise_t=ynoise_t
        if cynoise_e is None: cynoise_e=ynoise_e
        if cynoise_b is None: cynoise_b=ynoise_b
        
        return self._dict_from_theory_noise(self.theory2d,self.theory2d_norm,
                                xnoise_t,ynoise_t,
                                xnoise_e,ynoise_e,
                                xnoise_b,ynoise_b,
                                cxnoise_t,cynoise_t,
                                cxnoise_e,cynoise_e,
                                cxnoise_b,cynoise_b)                                
    def _dict_from_theory_noise(self,theory2d,theory2d_norm,
                                xnoise_t,ynoise_t,
                                xnoise_e,ynoise_e,
                                xnoise_b,ynoise_b,
                                cxnoise_t=None,cynoise_t=None,
                                cxnoise_e=None,cynoise_e=None,
                                cxnoise_b=None,cynoise_b=None):
        if cxnoise_t is None: cxnoise_t=xnoise_t
        if cxnoise_e is None: cxnoise_e=xnoise_e
        if cxnoise_b is None: cxnoise_b=xnoise_b
        if cynoise_t is None: cynoise_t=ynoise_t
        if cynoise_e is None: cynoise_e=ynoise_e
        if cynoise_b is None: cynoise_b=ynoise_b
        pol_list = ['TT','EE','TE','BB']
        xnoise = {'TT':xnoise_t,'EE':xnoise_e,'BB':xnoise_b,'TE':0}
        ynoise = {'TT':ynoise_t,'EE':ynoise_e,'BB':ynoise_b,'TE':0}
        cxnoise = {'TT':cxnoise_t,'EE':cxnoise_e,'BB':cxnoise_b,'TE':0}
        cynoise = {'TT':cynoise_t,'EE':cynoise_e,'BB':cynoise_b,'TE':0}
        fdict = {}
        for pol in pol_list:
            fdict['uClN_'+pol] = theory2d_norm['u'+pol]
            fdict['uCl_'+pol] = theory2d['u'+pol]
            fdict['tClX_'+pol] = theory2d['l'+pol] + xnoise[pol]
            fdict['tClY_'+pol] = theory2d['l'+pol] + ynoise[pol]
            fdict['tClcX_'+pol] = theory2d['l'+pol] + cxnoise[pol]
            fdict['tClcY_'+pol] = theory2d['l'+pol] + cynoise[pol]
                
        return fdict

    def cross(self,pol1,pol2,theory,xmask,ymask,noise_t=0,noise_e=0,noise_b=0,
              ynoise_t=None,ynoise_e=None,ynoise_b=None,
              cross_xnoise_t=None,cross_ynoise_t=None,
              cross_xnoise_e=None,cross_ynoise_e=None,
              cross_xnoise_b=None,cross_ynoise_b=None,
              theory_norm=None,hdv=True,save_expression="current",validate=True,cache=True,lensed_cls=None,xest=False,crossxest=False):
        if ynoise_t is None: ynoise_t = noise_t
        if ynoise_e is None: ynoise_e = noise_e
        if ynoise_b is None: ynoise_b = noise_b
        if cross_xnoise_t is None: cross_xnoise_t = noise_t
        if cross_xnoise_e is None: cross_xnoise_e = noise_e
        if cross_xnoise_b is None: cross_xnoise_b = noise_b
        if cross_ynoise_t is None: cross_ynoise_t = cross_xnoise_t
        if cross_ynoise_e is None: cross_ynoise_e = cross_xnoise_e
        if cross_ynoise_b is None: cross_ynoise_b = cross_xnoise_b
        uCl1 = self.Cls("uCl",self.l1)
        uCl2 = self.Cls("uCl",self.l2)
        tCl1 = self.Cls("tClX",self.l1)
        tCl2 = self.Cls("tClY",self.l2)
        if not(crossxest):
            tCl1b = tCl1
            tCl2b = tCl2
        else:
            tCl1b = self.Cls("tClY",self.l1)
            tCl2b = self.Cls("tClX",self.l2)
            
        Falpha = self.F_HDV(pol1,tCl1,tCl2,uCl1) if hdv else self.F_HuOk(pol1,tCl1,tCl2,uCl1,uCl2)
        Fbeta = self.F_HDV(pol2,tCl1b,tCl2b,uCl1) if hdv else self.F_HuOk(pol2,tCl1b,tCl2b,uCl1,uCl2)
        rFbeta = self.F_HDV(pol2,tCl1b,tCl2b,uCl1,uCl2,rev=True) if hdv else self.F_HuOk(pol2,tCl1b,tCl2b,uCl1,uCl2,rev=True)

        def sanitize(plist):
            return ['TE' if x=='ET' else x for x in plist]
        
        zero_list = ['TB','BT','EB','BE']
        Xa,Ya = pol1
        Xb,Yb = pol2
        XaXb,YaYb,XaYb,YaXb = sanitize([Xa+Xb,Ya+Yb,Xa+Yb,Ya+Xb])
        xestx = 'tClX_' if (xest and not(crossxest)) else 'tClcX_'
        xesty = 'tClY_' if (xest and not(crossxest)) else 'tClcY_'
        xestx2 = 'tClcX_' if (xest and not(crossxest)) else 'tClX_'
        xesty2 = 'tClcY_' if (xest and not(crossxest)) else 'tClY_'
        Cxaxb1 = self.f_of_ell(xestx+XaXb,self.l1) if XaXb not in zero_list else 0
        Cyayb2 = self.f_of_ell(xesty+YaYb,self.l2) if YaYb not in zero_list else 0
        Cxayb1 = self.f_of_ell(xestx2+XaYb,self.l1) if XaYb not in zero_list else 0
        Cyaxb2 = self.f_of_ell(xesty2+YaXb,self.l2) if YaXb not in zero_list else 0
        
        self.add_cross(save_expression,Falpha,Fbeta,rFbeta,Cxaxb1,Cyayb2,Cxayb1,Cyaxb2,validate=validate)

        theory2d,theory2d_norm = self._get_theory2d(theory,theory_norm,lensed_cls)
        feed_dict = self._dict_from_theory_noise(theory2d,theory2d_norm,
                                                 noise_t,ynoise_t,
                                                 noise_e,ynoise_e,
                                                 noise_b,ynoise_b,
                                                 cross_xnoise_t,cross_ynoise_t,
                                                 cross_xnoise_e,cross_ynoise_e,
                                                 cross_xnoise_b,cross_ynoise_b)
        cval = self.integrate(save_expression,feed_dict,xmask=xmask,ymask=ymask,cache=cache).real
        return cval

    def NL(self,AL=None,AL2=None,cross=None):
        if cross is None:
            assert AL2 is None
            return self.NL_from_AL(AL)
        else:
            return 0.25*(AL*AL2)*cross
    

    def f_K2(self,polcomb,uCl1,uCl2,rev=False):
        Ldl1 = self.Ldl1 if not(rev) else self.Ldl2
        Ldl2 = self.Ldl2 if not(rev) else self.Ldl1
        u1 = uCl1 if not(rev) else uCl2
        u2 = uCl2 if not(rev) else uCl1
        if polcomb=='TT':
            return Ldl1*(u1['TT']-u2['TT'])
        else:
            raise NotImplementedError

    def add_estimator(self,tag,f,F,feed_dict,xmask,ymask,validate=True,cache=True,norm_groups=None,est_groups=None):
        self.add_ALinv(tag,f,F,validate=validate,groups=norm_groups)
        X1 = self.f_of_ell('X1',self.l1)
        Y2 = self.f_of_ell('Y2',self.l2)
        self.Als[tag] = self.get_AL(tag,feed_dict,xmask,ymask,cache)
        expr = 0.5*F*X1*Y2
        self.add_factorized("_est_"+tag,expr,validate=validate,groups=est_groups)

    def qest(self,tag,feed_dict,xmask,ymask,kmask,cache=True):
        unnormalized = self.integrate("_est_"+tag,feed_dict,xmask=xmask,ymask=ymask,cache=cache,pixel_units=True)
        assert not(np.any(np.isnan(unnormalized)))
        res = np.nan_to_num(self.Als[tag] * unnormalized)
        res = mask_func(res,kmask)
        return res

    def add_maps(self,feed_dict,xmap,ymap=None):
        x1 = self._fft(xmap)
        y2 = self._fft(ymap) if not(ymap is None) else x1
        feed_dict['X1'] = x1
        feed_dict['Y2'] = y2

    def f_shear(self,uCl1,uCl2,rev=False):
        Ldl2 = self.Ldl1 if rev else self.Ldl2
        uC2 = uCl1['TT'] if rev else uCl2['TT']
        return Ldl2*uC2

    

    def F_shear(self,tCl1,tCl2,uCl1,duCl1,uCl2=None,duCl2=None,rev=False):
        Ldl1 = self.Ldl1 if not(rev) else self.Ldl2
        l1 = self.l1 if not(rev) else self.l2
        u1 = uCl1['TT'] if not(rev) else uCl2['TT']
        d1 = duCl1 if not(rev) else duCl2
        t1 = tCl1['TT'] if not(rev) else tCl2['TT']
        t2 = tCl2['TT'] if not(rev) else tCl1['TT']
        cos2theta = ((2*(Ldl1)**2)/self.L**2/l1**2) - 1
        return cos2theta * u1 * d1/2/t1/t1 # !!! tCl2??

    def Nl_cross_XXXX(self,tCl1,tCl2,F,Frev,X='T'):
        return tCl1[X+X]*tCl2[X+X]*F*(F+Frev)
    
    def Nl_shear_cross(self,uCl1,uCl2,tCl1,tCl2,duCl1,duCl2):
        F = self.F_shear(tCl1,tCl2,uCl1,uCl2=uCl2,duCl1=duCl1,duCl2=duCl2)
        Frev = self.F_shear(tCl1,tCl2,uCl1,uCl2=uCl2,duCl1=duCl1,duCl2=duCl2,rev=True)
        return self.Nl_cross_XXXX(tCl1,tCl2,F,Frev)
    
    def Nl_generic(self,Al,cross,feed_dict,xmask,ymask,cache=True,save_expression="current",validate=True,groups=None):
        self.add_factorized(save_expression,cross,validate=validate,groups=groups)
        ival = self.integrate(save_expression,feed_dict,xmask=xmask,ymask=ymask,cache=cache).real
        return np.nan_to_num(ival*Al**2.)/4.


        
        
def mask_func(arr,mask):
    arr[mask<1.e-3] = 0.
    return arr

def evaluate(symbolic_term,feed_dict):
    symbols = list(symbolic_term.free_symbols)
    func_term = sympy.lambdify(symbols,symbolic_term,dummify=False)
    # func_term accepts as keyword arguments strings that are in symbols
    # We need to extract a dict from feed_dict that only has the keywords
    # in symbols
    varstrs = [str(x) for x in symbols]
    edict = {k: feed_dict[k] for k in varstrs}
    evaled = np.nan_to_num(func_term(**edict))
    return evaled


def substitute_trig(l1x,l1y,l2x,l2y,l1,l2):
    phi1 = Symbol('phi1')
    phi2 = Symbol('phi2')
    cos2t12 = sympy.cos(2*(phi1-phi2))
    sin2t12 = sympy.sin(2*(phi1-phi2))
    simpcos = sympy.expand_trig(cos2t12)
    simpsin = sympy.expand_trig(sin2t12)
    cos2t12 = sympy.expand(sympy.simplify(simpcos.subs([(sympy.cos(phi1),l1x/l1),(sympy.cos(phi2),l2x/l2),
                            (sympy.sin(phi1),l1y/l1),(sympy.sin(phi2),l2y/l2)])))
    sin2t12 = sympy.expand(sympy.simplify(simpsin.subs([(sympy.cos(phi1),l1x/l1),(sympy.cos(phi2),l2x/l2),
                            (sympy.sin(phi1),l1y/l1),(sympy.sin(phi2),l2y/l2)])))
    return cos2t12,sin2t12
