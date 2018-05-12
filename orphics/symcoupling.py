from __future__ import print_function
import numpy as np
from sympy import Symbol,Function
import sympy
from enlib import fft as efft

"""
Routines to reduce and evaluate symbolic mode coupling integrals
"""

ifft = lambda x: efft.ifft(x,axes=[-2,-1],normalize=True)
fft = lambda x: efft.fft(x,axes=[-2,-1])

def factorize_2d_convolution_integral(expr,validate=True):
    """Reduce a sympy expression of variables l1x,l1y,l2x,l2y into a sum of 
    products of factors that depend only on vec(l1) and vec(l2). If the expression
    appeared as the integrand in an integral over vec(l1), where 
    vec(l2) = vec(L) - vec(l1) then this reduction allows one to evaluate the 
    integral as a function of vec(L) using FFTs instead of as a convolution.
    """

    # Generic message if validation fails
    val_fail_message = "Validation failed. This expression is likely not reducible to FFT form."

    # Get the 2D convolution cartesian variables
    l1x,l1y,l2x,l2y,l1,l2 = get_ells()

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
    for arg in arguments:
        # Lists to hold terms that depend on l1, l2 and neither
        ll1terms = []
        ll2terms = []
        loterms = []
        # Anticipate case where there's just a single factor
        argis = arg.args if len(arg.args)>=1 else [arg]

        # Loop through each factor in term, and group into l1, l2 and other
        for argi in argis:
            if argi.has(l1x) or argi.has(l1y) or argi.has(l1):
                ll1terms.append(argi)
            elif argi.has(l2x) or argi.has(l2y) or argi.has(l2):
                ll2terms.append(argi)
            else:
                loterms.append(argi)

        # Create a dictionary that holds the factorized terms
        tdict = {}
        tdict['l1'] = sympy.Mul(*ll1terms)
        tdict['l2'] = sympy.Mul(*ll2terms)
        tdict['other'] = sympy.Mul(*loterms)
        terms.append(tdict)

        # Validate!
        if validate:
            # Check that all the factors of this term do give back the original term
            products = sympy.Mul(*tdict.values())
            assert sympy.simplify(products-arg)==0, val_fail_message
            prodterms.append(products)
            # Check that the factors don't include symbols they shouldn't
            assert not(tdict['l1'].has(l2x)) and not(tdict['l2'].has(l1x)) and \
                not(tdict['other'].has(l1x)) and not(tdict['other'].has(l2x)) and \
                not(tdict['l1'].has(l2y)) and not(tdict['l2'].has(l1y)) and \
                not(tdict['other'].has(l1y)) and not(tdict['other'].has(l2y)) and \
                not(tdict['l1'].has(l2)) and not(tdict['l2'].has(l1)) and \
                not(tdict['other'].has(l1)) and not(tdict['other'].has(l2)), val_fail_message

    # Check that the sum of products of final form matches original expression
    if validate:
        fexpr = sympy.Add(*prodterms)
        assert sympy.simplify(expr-fexpr)==0, val_fail_message
    return terms


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
    return Lx,Ly
    
def qe_tt():

    Lx,Ly = get_Ls()
    l1x,l1y,l2x,l2y = get_ells()

    uC1 = Function('uC1')(l1x,l1y)
    uC2 = Function('uC1')(l2x,l2y)
    tC1 = Function('tC1')(l1x,l1y)
    tC2 = Function('tC2')(l2x,l2y)

    ldl1 = (Lx*l1x+Ly*l1y)
    ldl2 = (Lx*l2x+Ly*l2y)
    falpha = uC1*ldl1 + uC2*ldl2
    F = ldl1 * uC1 / tC1 / tC2

    expr = falpha*F
    terms = factorize_2d_convolution_integral(expr)
    for term in terms:
        print(term)


class ModeCoupling(object):

    def __init__(self):
        
        self.l1x,self.l1y,self.l2x,self.l2y,self.l1,self.l2 = get_ells()
        self.integrands = {}

    def add_factorized(self,tag,expr,validate=True):
        self.integrands[tag] = factorize_2d_convolution_integral(expr,validate=validate)

    def _evaluate(self,symbolic_term,feed_dict):
        symbols = list(symbolic_term.free_symbols)
        func_term = sympy.lambdify(symbols,symbolic_term,dummify=False)
        #func_term accepts as keyword arguments strings that are in symbols
        # We need to extract a dict from feed_dict that only has the keywords
        # in symbols
        varstrs = [str(x) for x in symbols]
        edict = {k: feed_dict[k] for k in varstrs}
        evaled = func_term(**edict)
        return evaled

    def eval(self,tag,feed_dict):
        shape = feed_dict['l1x'].shape
        ones = np.ones(shape,dtype=feed_dict['l1x'].dtype)
        val = 0.
        for term in self.integrands[tag]:
            l12d = self._evaluate(term['l1'],feed_dict)*ones
            l22d = self._evaluate(term['l2'],feed_dict)*ones
            ot2d = self._evaluate(term['other'],feed_dict)*ones
            val += ot2d*fft(ifft(l12d)*ifft(l22d))
        return val
        

class Lensing(ModeCoupling):
    def __init__(self):
        ModeCoupling.__init__(self)
        self.Lx,self.Ly = get_Ls()
        self.Ldl1 = (self.Lx*self.l1x+self.Ly*self.l1y)
        self.Ldl2 = (self.Lx*self.l2x+self.Ly*self.l2y)

    def F_HuOk(self):
        pass


    def f(self,polcomb):
        pass
        

    def add_AL(self,tag,fa,Fa,validate=True):
        expr = fa*Fa
        self.add_factorized(tag,expr,validate=validate)    
        
    def add_cross(self,tag,Fa,Fb,Fbr,Cxaxb,Cyayb,Cxayb,Cyaxb,validate=True):
        # integrand in Eq 17 in HuOk01
        expr = Fa*(Fb*Cxaxb*Cyayb+Fbr*Cxayb*Cyaxb)
        self.add_factorized(tag,expr,validate=validate)    
                           
mc = ModeCoupling()
mc.add_factorized("test",mc.l1x,validate=True)
shape = (100,100)
l1x = np.ones(shape,dtype=np.complex128)
mc.eval("test",{'l1x':l1x,'l2x':3})
