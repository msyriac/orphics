from __future__ import print_function
import numpy as np
from sympy import Symbol,Function
import sympy
from enlib import fft as efft, enmap, bench
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
                if found: raise ValueError, "Groups don't seem to be mutually exclusive."
                index = i
                found = True
        if not(found): raise ValueError, "Couldn't associate a group"
        return index


    ogroups = [] if not(groups is None) else None
    ogroup_weights = [] if not(groups is None) else None
    ogroup_symbols = sympy.ones(len(groups),1) if not(groups is None) else None
    for k,arg in enumerate(arguments):
        temp, ll1terms = arg.as_independent(*l1funcs, as_Mul=True)
        loterms, ll2terms = temp.as_independent(*l2funcs, as_Mul=True)

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
        if groups is None: groups = self._default_groups
        self.integrands[tag],self.ul1s[tag],self.ul2s[tag], \
            self.ogroups[tag],self.ogroup_weights[tag], \
            self.ogroup_symbols[tag] = factorize_2d_convolution_integral(expr,l1funcs=self.l1funcs,l2funcs=self.l2funcs,
                                                                         validate=validate,groups=groups)

    def _evaluate(self,symbolic_term,feed_dict):
        symbols = list(symbolic_term.free_symbols)
        func_term = sympy.lambdify(symbols,symbolic_term,dummify=False)
        # func_term accepts as keyword arguments strings that are in symbols
        # We need to extract a dict from feed_dict that only has the keywords
        # in symbols
        varstrs = [str(x) for x in symbols]
        edict = {k: feed_dict[k] for k in varstrs}
        evaled = np.nan_to_num(func_term(**edict))
        return evaled

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
                l12d = self._evaluate(u1,feed_dict)*ones
                cached_u1s.append(self._ifft(l12d*xmask))
            for u2 in self.ul2s[tag]:
                l22d = self._evaluate(u2,feed_dict)*ones
                cached_u2s.append(self._ifft(l22d*ymask))


        # For each term, the index of which group it belongs to  
        ogroups = self.ogroups[tag] 
        ogroup_weights = self.ogroup_weights[tag] 
        ogroup_symbols = self.ogroup_symbols[tag]

        
        if ogroups is None:    
            for i,term in enumerate(self.integrands[tag]):
                if cache:
                    ifft1 = cached_u1s[term['l1index']]
                    ifft2 = cached_u2s[term['l2index']]
                else:
                    l12d = self._evaluate(term['l1'],feed_dict)*ones
                    ifft1 = self._ifft(l12d*xmask)
                    l22d = self._evaluate(term['l2'],feed_dict)*ones
                    ifft2 = self._ifft(l22d*ymask)
                ot2d = self._evaluate(term['other'],feed_dict)*ones
                ffft = self._fft(ifft1*ifft2)
                val += ot2d*ffft.real
        else:
            vals = np.zeros((len(ogroup_symbols),)+shape,dtype=feed_dict['L'].dtype)+0j
            for i,term in enumerate(self.integrands[tag]):
                if cache:
                    ifft1 = cached_u1s[term['l1index']]
                    ifft2 = cached_u2s[term['l2index']]
                else:
                    l12d = self._evaluate(term['l1'],feed_dict)*ones
                    ifft1 = self._ifft(l12d*xmask)
                    l22d = self._evaluate(term['l2'],feed_dict)*ones
                    ifft2 = self._ifft(l22d*ymask)
                gindex = ogroups[i]
                vals[gindex,...] += ifft1*ifft2 *ogroup_weights[i]
            for i,group in enumerate(ogroup_symbols):
                ot2d = self._evaluate(ogroup_symbols[i],feed_dict)*ones            
                ffft = self._fft(vals[i,...])
                val += ot2d*ffft.real

                
        mul = 1 if pixel_units else 1./self.pixarea
        return val * mul

    def _fft(self,x):
        self.nfft += 1
        return fft(x+0j)
    def _ifft(self,x):
        self.nifft += 1
        return ifft(x+0j)
        

class LensingModeCoupling(ModeCoupling):
    def __init__(self,shape,wcs):
        ModeCoupling.__init__(self,shape,wcs)
        self.Lx,self.Ly,self.L = get_Ls()
        self.Ldl1 = (self.Lx*self.l1x+self.Ly*self.l1y)
        self.Ldl2 = (self.Lx*self.l2x+self.Ly*self.l2y)
        self.l1dl2 = (self.l1x*self.l2x+self.l2x*self.l2y)

        phi1 = Symbol('phi1')
        phi2 = Symbol('phi2')
        cos2t12 = sympy.cos(2*(phi1-phi2))
        sin2t12 = sympy.sin(2*(phi1-phi2))
        simpcos = sympy.expand_trig(cos2t12)
        simpsin = sympy.expand_trig(sin2t12)

        self.cos2t12 = sympy.expand(sympy.simplify(simpcos.subs([(sympy.cos(phi1),self.l1x/self.l1),(sympy.cos(phi2),self.l2x/self.l2),
                                (sympy.sin(phi1),self.l1y/self.l1),(sympy.sin(phi2),self.l2y/self.l2)])))

        self.sin2t12 = sympy.expand(sympy.simplify(simpsin.subs([(sympy.cos(phi1),self.l1x/self.l1),(sympy.cos(phi2),self.l2x/self.l2),
                                (sympy.sin(phi1),self.l1y/self.l1),(sympy.sin(phi2),self.l2y/self.l2)])))



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

    def F_HuOk(self,polcomb,tCl1,tCl2,uCl1,uCl2,rev=False):
        t1 = tCl1 if not(rev) else tCl2
        t2 = tCl2 if not(rev) else tCl1
        
        if polcomb=='TE':
            f = self.f(polcomb,uCl1,uCl2,rev=rev)
            frev = self.f(polcomb,uCl1,uCl2,rev=not(rev))
            # this filter is not separable
            #(tCl1['EE']*tCl2['TT']*f - tCl1['TE']*tCl2['TE']*frev)/(tCl1['TT']*tCl2['EE']*tCl1['EE']*tCl2['TT']-(tCl1['TE']*tCl2['TE'])**2.)
            # this approximation is
            return (t1['EE']*t2['TT']*f - t1['TE']*t2['TE']*frev)/(t1['TT']*t2['EE']*t1['EE']*t2['TT'])
            
        X,Y = polcomb
        pfact = 0.5 if Y!='B' else 1.0
        f = self.f(polcomb,uCl1,uCl2,rev=rev)
        return f*pfact/t1[X+X]/t2[Y+Y]
        

        
    def F_HDV(self,polcomb,tCl1,tCl2,uCl1):
        X,Y = polcomb
        Ldl1 = self.Ldl1
        pref = Ldl1/tCl1[X+X]/tCl2[Y+Y]
        if Y=='T':
            return pref * uCl1[X+Y]
        if Y=='E':
            return pref * uCl1[X+Y] * self.cos2t12
        if Y=='B':
            return pref * uCl1[X+'E'] * self.sin2t12
            
        
    def add_ALinv(self,tag,fa,Fa,validate=True):
        expr = fa*Fa/self.L**2
        self.add_factorized(tag,expr,validate=validate)

    def get_AL(self,tag,feed_dict,xmask=None,ymask=None,cache=True):
        ival = mc.integrate(tag,feed_dict,xmask=xmask,ymask=ymask,cache=cache)
        return np.nan_to_num(1./ival)

    def NL_from_AL(self,AL):
        return AL*self.modlmap**2./4.
        
    def get_NL(self,tag,feed_dict,xmask=None,ymask=None,cache=True):
        AL = self.get_AL(tag,feed_dict,xmask=xmask,ymask=xmask,cache=cache)
        return self.NL_from_AL(AL)
        
    def add_cross(self,tag,Fa,Fb,Fbr,Cxaxb1,Cyayb2,Cxayb1,Cyaxb2,validate=True):
        # integrand in Eq 17 in HuOk01
        expr = Fa*(Fb*Cxaxb1*Cyayb2+Fbr*Cxayb1*Cyaxb2)
        self.add_factorized(tag,expr,validate=validate)    


from orphics import maps
cache = True
deg = 5
px = 1.0
shape,wcs = maps.rect_geometry(width_deg = deg,px_res_arcmin=px)
mc = LensingModeCoupling(shape,wcs)
uCl1 = mc.Cls("uCl",mc.l1)
uCl2 = mc.Cls("uCl",mc.l2)
tCl1 = mc.Cls("tCl",mc.l1)
tCl2 = mc.Cls("tCl",mc.l2)
pol = "EE"
f = mc.f(pol,uCl1,uCl2)
F = mc.F_HuOk(pol,tCl1,tCl2,uCl1,uCl2)
Frev = mc.F_HuOk(pol,tCl1,tCl2,uCl1,uCl2,rev=True)
#F = mc.F_HDV(pol,tCl1,tCl2,uCl1)
with bench.show("eval"):
    mc.add_ALinv("test",f,F,validate=True)

# for t in mc.integrands['test']:
#     print(t['l1'])
#     print(t['l2'])
#     print(t['other'])
#     print("----")
# print(len(mc.integrands['test']))

theory = cosmology.default_theory(lpad=20000)
noise_t = 27.0
noise_p = 40.0*np.sqrt(2.)
fwhm = 7.0
kbeam = maps.gauss_beam(fwhm,mc.modlmap)
ells = np.arange(0,3000,1)
lbeam = maps.gauss_beam(fwhm,ells)
ntt = np.nan_to_num((noise_t*np.pi/180./60.)**2./kbeam**2.)
nee = np.nan_to_num((noise_p*np.pi/180./60.)**2./kbeam**2.)
nbb = np.nan_to_num((noise_p*np.pi/180./60.)**2./kbeam**2.)
lntt = np.nan_to_num((noise_t*np.pi/180./60.)**2./lbeam**2.)
lnee = np.nan_to_num((noise_p*np.pi/180./60.)**2./lbeam**2.)
lnbb = np.nan_to_num((noise_p*np.pi/180./60.)**2./lbeam**2.)


uclee = theory.uCl('EE',mc.modlmap)
uclte = theory.uCl('TE',mc.modlmap)
tclee = theory.lCl('EE',mc.modlmap) + nee
tclbb = theory.lCl('BB',mc.modlmap) + nbb
tcltt = theory.lCl('TT',mc.modlmap) + ntt
tclte = theory.lCl('TE',mc.modlmap) 
ucltt = theory.uCl('TT',mc.modlmap)

ellmin = 20
ellmax = 3000
xmask = maps.mask_kspace(shape,wcs,lmin=ellmin,lmax=ellmax)
ymask = xmask

with bench.show("ALcalc"):
    AL = mc.get_AL("test",{'uCl_EE':uclee,'tCl_EE':tclee,'tCl_BB':tclbb,'tCl_TT':tcltt,'uCl_TT':ucltt,'uCl_TE':uclte,'tCl_TE':tclte},xmask=xmask,ymask=ymask,cache=cache)
val = mc.NL_from_AL(AL)

bin_edges = np.arange(10,2000,40)
cents,nkk = stats.bin_in_annuli(val,mc.modlmap,bin_edges)

ls,hunls = np.loadtxt("alhazen/data/hu_"+pol.lower()+".csv",delimiter=',',unpack=True)
pl = io.Plotter(yscale='log')
pl.add(ells,theory.gCl('kk',ells),lw=3,color='k')
pl.add(cents,nkk,ls="--")
pl.add(ls,hunls*2.*np.pi/4.,ls="-.")

oest = ['TE','ET'] if pol=='TE' else [pol]
ls,nlkks,theory,qest = lensing.lensing_noise(ells,lntt,lnee,lnbb,
                  ellmin,ellmin,ellmin,
                  ellmax,ellmax,ellmax,
                  bin_edges,
                  theory=theory,
                  estimators = oest,
                  unlensed_equals_lensed=False,
                  width_deg=10.,px_res_arcmin=1.0)
    
pl.add(ls,nlkks['mv'],ls="-")


# mc.add_cross("cross",F,F,Frev,tCl1['TT'],tCl2['TT'],tCl1['TT'],tCl2['TT'],validate=True)
# cval = mc.integrate("cross",{'uCl_EE':uclee,'tCl_EE':tclee,'tCl_BB':tclbb,'tCl_TT':tcltt,'uCl_TT':ucltt,'uCl_TE':uclte,'tCl_TE':tclte},xmask=xmask,ymask=ymask,cache=cache)

# Nlalt = 0.25*(AL**2.)*cval
# cents,nkkalt = stats.bin_in_annuli(Nlalt,mc.modlmap,bin_edges)
# pl.add(cents,nkkalt,marker="o",alpha=0.2)

pl.done()


print("nffts : ",mc.nfft,mc.nifft)
