from __future__ import print_function
import numpy as np
from sympy import Symbol,Function
import sympy
from enlib import fft as efft, enmap
from orphics import maps,io,stats,cosmology,lensing

"""
Routines to reduce and evaluate symbolic mode coupling integrals
"""

ifft = lambda x: efft.ifft(x,axes=[-2,-1],normalize=True)
fft = lambda x: efft.fft(x,axes=[-2,-1])

def factorize_2d_convolution_integral(expr,l1funcs=None,l2funcs=None,validate=True):
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
    Lx = sympy.Symbol('Lx')
    Ly = sympy.Symbol('Ly')
    L = sympy.Symbol('L')
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
    unique_l1l2s = []
    
    def homogenize(inexp):
        outexp = inexp.subs([[l1x,Lx],[l2x,Lx],[l1y,Ly],[l2y,Ly],[l1,L],[l2,L]])
        ofuncs = ofuncs1.union(ofuncs2)
        for ofunc in ofuncs:
            nfunc = sympy.Symbol(str(ofunc)[:-3])
            outexp = outexp.subs(ofunc,nfunc)
        return outexp
        
        
    
    for arg in arguments:
        # Lists to hold terms that depend on l1, l2 and neither
        ll1terms = []
        ll2terms = []
        loterms = []
        # Anticipate case where there's just a single factor
        argis = arg.args if len(arg.args)>=1 else [arg]
        # Loop through each factor in term, and group into l1, l2 and other
        for argi in argis:
            if any([argi.has(x) for x in l1funcs]):
                ll1terms.append(argi)
            elif any([argi.has(x) for x in l2funcs]):
                ll2terms.append(argi)
            else:
                loterms.append(argi)
        # Create a dictionary that holds the factorized terms
        vdict = {}
        vdict['l1'] = sympy.Mul(*ll1terms)
        vdict['l2'] = sympy.Mul(*ll2terms)
        tdict = {}
        tdict['l1'] = homogenize(vdict['l1'])
        tdict['l2'] = homogenize(vdict['l2'])
        tdict['l1l2'] = tdict['l1']*tdict['l2']

        if not(tdict['l1'] in unique_l1s):
            unique_l1s.append(tdict['l1'])
        tdict['l1index'] = unique_l1s.index(tdict['l1'])

        if not(tdict['l2'] in unique_l2s):
            unique_l2s.append(tdict['l2'])
        tdict['l2index'] = unique_l2s.index(tdict['l2'])

        if not(tdict['l1l2'] in unique_l1l2s):
            unique_l1l2s.append(tdict['l1l2'])
        tdict['l1l2index'] = unique_l1l2s.index(tdict['l1l2'])
        
        
        vdict['other'] = sympy.Mul(*loterms)
        tdict['other'] = sympy.Mul(*loterms)
        terms.append(tdict)
        # Validate!
        if validate:
            # Check that all the factors of this term do give back the original term
            products = sympy.Mul(vdict['l1'])*sympy.Mul(vdict['l2'])*sympy.Mul(vdict['other'])
            assert sympy.simplify(products-arg)==0, val_fail_message
            prodterms.append(products)
            # Check that the factors don't include symbols they shouldn't
            assert all([not(vdict['l1'].has(x)) for x in l2funcs])
            assert all([not(vdict['l2'].has(x)) for x in l1funcs])
            assert all([not(vdict['other'].has(x)) for x in l1funcs])
            assert all([not(vdict['other'].has(x)) for x in l2funcs])
    # Check that the sum of products of final form matches original expression
    if validate:
        fexpr = sympy.Add(*prodterms)
        assert sympy.simplify(expr-fexpr)==0, val_fail_message
    return terms,unique_l1s,unique_l2s,unique_l1l2s


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

    def __init__(self,shape,wcs):
        # Symbolic
        self.l1x,self.l1y,self.l2x,self.l2y,self.l1,self.l2 = get_ells()
        self.integrands = {}
        self.l1funcs = []
        self.l2funcs = []
        # Diagnostic
        self.nfft = 0
        self.nifft = 0
        # Numeric
        self.shape,self.wcs = shape,wcs
        self.modlmap = enmap.modlmap(shape,wcs)
        self.lymap,self.lxmap = enmap.lmap(shape,wcs)

    def f_of_ell(self,name,ell):
        fname = name+"_"+str(ell)
        f = Symbol(fname)
        if ell in [self.l1x,self.l1y,self.l1]: self.l1funcs.append(f)
        if ell in [self.l2x,self.l2y,self.l2]: self.l2funcs.append(f)
        return f
    
    def add_factorized(self,tag,expr,validate=True):
        self.integrands[tag],self.ul1s,self.ul2s,self.ul1l2s = factorize_2d_convolution_integral(expr,
                                                                                     l1funcs=self.l1funcs,l2funcs=self.l2funcs,validate=validate)

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

    def integrate(self,tag,feed_dict,xmask=None,ymask=None,cache=True):
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
            for u1 in self.ul1s:
                l12d = self._evaluate(u1,feed_dict)*ones
                cached_u1s.append(self._ifft(l12d*xmask))
            for u2 in self.ul2s:
                l22d = self._evaluate(u2,feed_dict)*ones
                cached_u2s.append(self._ifft(l22d*ymask))
                
            print("u1 ",len(cached_u1s))
            print("u2 ",len(cached_u2s))
            
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
        return val

    def _fft(self,x):
        self.nfft += 1
        return fft(x+0j)
    def _ifft(self,x):
        self.nifft += 1
        return ifft(x+0j)
        

class Lensing(ModeCoupling):
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
        
    def f(self,polcomb,uCl1,uCl2):
        if polcomb=='TT':
            return self.Ldl1*uCl1['TT']+self.Ldl2*uCl2['TT']
        elif polcomb=='TE':
            return self.Ldl1*self.cos2t12*uCl1['TE']+self.Ldl2*uCl2['TE']
        elif polcomb=='ET':
            return self.Ldl2*uCl2['TE']*self.cos2t12+self.Ldl1*uCl1['TE']
        elif polcomb=='TB':
            return uCl1['TE']*self.sin2t12*self.Ldl1
        elif polcomb=='EE':
            return (self.Ldl1*uCl1['EE']+self.Ldl2*uCl2['EE'])*self.cos2t12
        elif polcomb=='EB':
            return self.Ldl1*uCl1['EE']*self.sin2t12

    def F_HuOk(self,polcomb,tCl1,tCl2,uCl1,uCl2):
        if polcomb=='TE' or polcomb=='ET': raise NotImplementedError
        X,Y = polcomb
        pfact = 0.5 if Y!='B' else 1.0
        f = self.f(polcomb,uCl1,uCl2)
        return f*pfact/tCl1[X+X]/tCl2[Y+Y]
        

        
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
        expr = fa*Fa
        self.add_factorized(tag,expr,validate=validate)    
        
    def add_cross(self,tag,Fa,Fb,Fbr,Cxaxb,Cyayb,Cxayb,Cyaxb,validate=True):
        # integrand in Eq 17 in HuOk01
        expr = Fa*(Fb*Cxaxb*Cyayb+Fbr*Cxayb*Cyaxb)
        self.add_factorized(tag,expr,validate=validate)    


from orphics import maps
cache = True
deg = 10
px = 1.0
shape,wcs = maps.rect_geometry(width_deg = deg,px_res_arcmin=px)
mc = Lensing(shape,wcs)
uCl1 = mc.Cls("uCl",mc.l1)
uCl2 = mc.Cls("uCl",mc.l2)
tCl1 = mc.Cls("tCl",mc.l1)
tCl2 = mc.Cls("tCl",mc.l2)
pol = "TT"
#mc.add_ALinv("test",mc.f(pol,uCl1,uCl2),mc.F_HDV(pol,tCl1,tCl2,uCl1),validate=True)
mc.add_ALinv("test",mc.f(pol,uCl1,uCl2),mc.F_HuOk(pol,tCl1,tCl2,uCl1,uCl2),validate=True)

# for t in mc.integrands['test']:
#     print(t['l1'])
#     print(t['l2'])
#     print(t['other'])
#     print("----")
# print(len(mc.integrands['test']))

theory = cosmology.default_theory(lpad=20000)
noise = 27.0
fwhm = 7.0
kbeam = maps.gauss_beam(fwhm,mc.modlmap)
ells = np.arange(0,3000,1)
lbeam = maps.gauss_beam(fwhm,ells)
ntt = np.nan_to_num((noise*np.pi/180./60.)**2./kbeam**2.)
lntt = np.nan_to_num((noise*np.pi/180./60.)**2./lbeam**2.)



uclee = theory.uCl('EE',mc.modlmap)
tclee = theory.lCl('EE',mc.modlmap)
tclbb = theory.lCl('BB',mc.modlmap)
tcltt = theory.lCl('TT',mc.modlmap) + ntt
ucltt = theory.uCl('TT',mc.modlmap)

ellmin = 2
ellmax = 3000
xmask = maps.mask_kspace(shape,wcs,lmin=ellmin,lmax=ellmax)
ymask = xmask

ival = mc.integrate("test",{'uCl_EE':uclee,'tCl_EE':tclee,'tCl_BB':tclbb,'tCl_TT':tcltt,'uCl_TT':ucltt},xmask=xmask,ymask=ymask,cache=cache)
pixScaleY,pixScaleX = enmap.pixshape(shape,wcs)
val = np.nan_to_num(mc.modlmap**4./ival/4.* pixScaleX*pixScaleY  )
#io.plot_img(np.fft.fftshift(val))

bin_edges = np.arange(10,2000,40)
cents,nkk = stats.bin_in_annuli(val,mc.modlmap,bin_edges)

ls,hunls = np.loadtxt("alhazen/data/hu_tt.csv",delimiter=',',unpack=True)
pl = io.Plotter(yscale='log')
pl.add(ells,theory.gCl('kk',ells),lw=3,color='k')
pl.add(cents,nkk,ls="--")
pl.add(ls,hunls*2.*np.pi/4.,ls="-.")


ls,nlkks,theory,qest = lensing.lensing_noise(ells,lntt,lntt*0,lntt*0,
                  ellmin,ellmin,ellmin,
                  ellmax,ellmax,ellmax,
                  bin_edges,
                  theory=theory,
                  estimators = ['TT'],
                  unlensed_equals_lensed=False,
                  width_deg=10.,px_res_arcmin=1.0)
    
pl.add(ls,nlkks['TT'],ls="-")

pl.done()


print("nffts : ",mc.nfft,mc.nifft)
print("res : ", val)
